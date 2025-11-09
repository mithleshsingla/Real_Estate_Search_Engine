"""
Fixed LangGraph Multi-Agent System
Properly handles agent retries and prevents infinite loops
"""
from datetime import datetime
import uuid
import logging
from typing import Any, Dict, Tuple
import os

# LangGraph imports
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    try:
        from langgraph.checkpoint import SqliteSaver
    except ImportError:
        SqliteSaver = None

from langgraph.graph import StateGraph, END
from state import AgentState, IntentType
from router import router_node
from planner import planner_node
from sql_agent import sql_agent_node
from rag_agent import rag_agent_node
from web_agent import web_agent_node
from renovation_agent import renovation_agent_node
from report_agent import report_agent_node, memory_node

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multiagent.graph")

# Configuration
MAX_AGENT_RETRIES = 3  # Maximum times to retry an agent before moving on


def should_continue_tasks(state: AgentState) -> str:
    """
    Determine next agent to call based on tasks and execution status
    FIXED: Properly checks agent call counts and moves to next task
    """
    tasks = state.get("tasks", []) or []
    current_idx = state.get("current_task_index", 0)
    agent_call_count = state.get("agent_call_count", {})
    max_retries = state.get("max_agent_retries", MAX_AGENT_RETRIES)
    
    # CRITICAL FIX: Don't modify task list - it causes exponential growth
    # Just work with the original task list
    
    # If current task is marked completed, move to next
    if state.get("task_completed", False):
        current_idx += 1
        state["current_task_index"] = current_idx
        state["task_completed"] = False  # Reset flag
    
    # No more tasks - go to aggregator
    if current_idx >= len(tasks):
        logger.info("✓ All tasks completed, moving to aggregator")
        return "aggregate"
    
    current_task = tasks[current_idx]
    logger.info(f"Processing task {current_idx + 1}/{len(tasks)}: {current_task}")
    
    # Determine agent for current task
    task_upper = str(current_task).upper()
    
    if "SQL" in task_upper:
        agent_name = "sql_agent"
    elif "RAG" in task_upper:
        agent_name = "rag_agent"
    elif "WEB" in task_upper:
        agent_name = "web_agent"
    elif "RENOVATION" in task_upper:
        agent_name = "renovation_agent"
    elif "REPORT" in task_upper:
        agent_name = "report_agent"
    else:
        # Unknown task, skip to next
        logger.warning(f"Unknown task type: {current_task}, skipping")
        state["current_task_index"] = current_idx + 1
        return should_continue_tasks(state)
    
    # Check if agent has exceeded retry limit
    call_count = agent_call_count.get(agent_name, 0)
    
    if call_count >= max_retries:
        logger.warning(f"⚠️ {agent_name} exceeded {max_retries} retries, skipping task")
        # Move to next task
        state["current_task_index"] = current_idx + 1
        # Recursively check next task
        return should_continue_tasks(state)
    
    logger.info(f"→ Routing to {agent_name} (attempt {call_count + 1}/{max_retries})")
    return agent_name


def increment_agent_count(state: AgentState, agent_name: str) -> AgentState:
    """Helper to increment agent call counter"""
    if "agent_call_count" not in state:
        state["agent_call_count"] = {}
    
    state["agent_call_count"][agent_name] = state["agent_call_count"].get(agent_name, 0) + 1
    return state


def wrap_agent_with_retry_check(agent_func, agent_name: str):
    """
    Wraps agent function to:
    1. Increment call counter
    2. Mark task as completed if results are successful
    3. Handle failures gracefully
    """
    def wrapped(state: AgentState) -> AgentState:
        # Increment counter
        state = increment_agent_count(state, agent_name)
        
        try:
            # Call original agent
            state = agent_func(state)
            
            # Check if agent produced results
            has_results = False
            
            if agent_name == "sql_agent":
                sql_results = state.get("sql_results")
                has_results = sql_results and sql_results.get("count", 0) > 0
            
            elif agent_name == "rag_agent":
                rag_results = state.get("rag_results")
                has_results = rag_results and rag_results.get("answer")
            
            elif agent_name == "web_agent":
                web_results = state.get("web_results")
                has_results = web_results and web_results.get("insights")
            
            elif agent_name == "renovation_agent":
                ren_results = state.get("renovation_estimate")
                has_results = ren_results and "total_cost" in ren_results
            
            elif agent_name == "report_agent":
                has_results = state.get("report_path") is not None
            
            # If results found, mark task as completed
            if has_results:
                logger.info(f"✓ {agent_name} produced results, marking task complete")
                state["task_completed"] = True
            else:
                logger.warning(f"⚠️ {agent_name} produced no results, will retry or skip")
                state["task_completed"] = False
        
        except Exception as e:
            logger.error(f"✗ {agent_name} failed: {e}")
            state["errors"].append(f"{agent_name}: {str(e)}")
            state["task_completed"] = False
        
        return state
    
    return wrapped

def aggregate_results(state: AgentState) -> AgentState:
    """Combine partial results into a unified final response, with LLM/Tavily fallback."""
    import httpx

    parts = []

    sql = state.get("sql_results")
    if sql and isinstance(sql, dict) and sql.get("count", 0) > 0:
        parts.append(f"Found {sql['count']} properties matching your criteria.")

    rag = state.get("rag_results")
    if rag and rag.get("answer"):
        parts.append(rag["answer"])

    web = state.get("web_results")
    if web and web.get("insights"):
        parts.append(f"\n**Market Insights:**\n{web['insights']}")

    ren = state.get("renovation_estimate")
    if ren and "total_cost" in ren:
        parts.append(f"\n**Renovation Estimate:** ₹{ren['total_cost']:,}")

    if state.get("report_path"):
        parts.append(f"\n**Report Generated:** {state['report_path']}")

    # --- 🌐 FALLBACK SECTION ---
    if not parts:
        query = state.get("user_query", "Unknown query")
        logger.warning("⚠️ No agent produced results — triggering fallback search")

        try:
            # First try Tavily search
            tavily_key = os.getenv("TAVILY_API_KEY")
            tavily_results = []
            if tavily_key:
                r = httpx.post(
                    "https://api.tavily.com/search",
                    json={"query": query, "num_results": 3},
                    headers={"Authorization": f"Bearer {tavily_key}"}
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("results"):
                        tavily_results = [item["content"] for item in data["results"][:3]]
                        parts.append("**Web Insights (via Tavily):**\n" + "\n".join(tavily_results))
                        state["web_results"] = {"insights": tavily_results}
                        state["executed_agents"].append("tavily_fallback")
            
            # If still empty, call LLM fallback
            if not parts:
                groq_key = os.getenv("GROQ_API_KEY")
                if groq_key:
                    r = httpx.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}"},
                        json={
                            "model": "llama-3-70b-8192",
                            "messages": [{"role": "user", "content": f"Answer this real estate query: {query}"}],
                            "temperature": 0.7,
                        }
                    )
                    if r.status_code == 200:
                        content = r.json()["choices"][0]["message"]["content"]
                        parts.append("**LLM Reasoning Fallback:**\n" + content)
                        state["llm_fallback_response"] = content
                        state["executed_agents"].append("llm_fallback")

        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            parts.append("No direct results found, and fallback search also failed.")

    # Combine all parts
    final = "\n\n".join(parts)
    cites = state.get("citations") or []
    if cites:
        final += "\n\n**References:**\n" + "\n".join([f"- {c}" for c in cites[:5]])

    state["final_response"] = final

    if "executed_agents" not in state:
        state["executed_agents"] = []
    state["executed_agents"].append("aggregator")

    logger.info(f"✓ Aggregated final response ({len(final)} chars)")
    return state

def create_graph(checkpoint_conn: str = ":memory:") -> Any:
    """Build the LangGraph workflow with proper retry logic"""
    wf = StateGraph(AgentState)
    
    # Wrap agents with retry checking
    wrapped_sql = wrap_agent_with_retry_check(sql_agent_node, "sql_agent")
    wrapped_rag = wrap_agent_with_retry_check(rag_agent_node, "rag_agent")
    wrapped_web = wrap_agent_with_retry_check(web_agent_node, "web_agent")
    wrapped_renovation = wrap_agent_with_retry_check(renovation_agent_node, "renovation_agent")
    wrapped_report = wrap_agent_with_retry_check(report_agent_node, "report_agent")
    
    # Add nodes
    wf.add_node("router", router_node)
    wf.add_node("planner", planner_node)
    wf.add_node("sql_agent", wrapped_sql)
    wf.add_node("rag_agent", wrapped_rag)
    wf.add_node("web_agent", wrapped_web)
    wf.add_node("renovation_agent", wrapped_renovation)
    wf.add_node("report_agent", wrapped_report)
    wf.add_node("aggregate", aggregate_results)
    wf.add_node("memory", memory_node)
    
    # Define edges
    wf.set_entry_point("router")
    wf.add_edge("router", "planner")
    
    # After planner, route to appropriate agent
    wf.add_conditional_edges(
        "planner",
        should_continue_tasks,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "web_agent": "web_agent",
            "renovation_agent": "renovation_agent",
            "report_agent": "report_agent",
            "aggregate": "aggregate"
        }
    )
    
    # After each agent, check if should continue to next task
    for agent in ["sql_agent", "rag_agent", "web_agent", "renovation_agent", "report_agent"]:
        wf.add_conditional_edges(
            agent,
            should_continue_tasks,
            {
                "sql_agent": "sql_agent",
                "rag_agent": "rag_agent",
                "web_agent": "web_agent",
                "renovation_agent": "renovation_agent",
                "report_agent": "report_agent",
                "aggregate": "aggregate"
            }
        )
    
    wf.add_edge("aggregate", "memory")
    wf.add_edge("memory", END)
    
    # Create checkpoint
    checkpointer = None
    if SqliteSaver:
        try:
            checkpointer = SqliteSaver.from_conn_string(checkpoint_conn)
        except Exception as e:
            logger.warning(f"Failed to init SqliteSaver: {e}")
    
    # Compile with increased recursion limit
    app = wf.compile(
        checkpointer=checkpointer,
        debug=False
    )
    
    return app


def run_agent_system(query: str, session_id: str = None) -> Tuple[str, Dict[str, Any]]:
    """Run the workflow end-to-end with proper initialization"""
    session_id = session_id or str(uuid.uuid4())
    
    state = {
        "user_query": query,
        "session_id": session_id,
        "timestamp": datetime.now(),
        "intent": None,
        "tasks": [],
        "current_task_index": 0,
        
        # Initialize retry tracking
        "agent_call_count": {},
        "max_agent_retries": MAX_AGENT_RETRIES,
        "task_completed": False,
        
        "sql_results": None,
        "rag_results": None,
        "web_results": None,
        "renovation_estimate": None,
        "report_path": None,
        "intermediate_responses": [],
        "final_response": "",
        "citations": [],
        "conversation_history": [],
        "user_preferences": {},
        "executed_agents": [],
        "errors": [],
        "confidence_score": None
    }
    
    app = create_graph()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 50  # Increase from default 25
    }
    
    logger.info(f"▶️ Running multi-agent workflow | session={session_id[:8]}... | query='{query}'")
    
    final_state = state.copy()
    step_count = 0

    try:
        for s in app.stream(state, config):
            step_count += 1
            if not isinstance(s, dict):
                continue

            for node_name, node_state in s.items():
                logger.info(f"Step {step_count}: {node_name}")
                if isinstance(node_state, dict):
                    final_state.update(node_state)
    
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return f"Error: {str(e)}", state
    
    if not final_state:
        logger.warning("No final state generated.")
        return "No response generated", {}
    
    logger.info(f"✅ Workflow completed in {step_count} steps")
    logger.info(f"Executed agents: {final_state.get('executed_agents', [])}")
    
    if final_state.get("errors"):
        logger.warning(f"Errors encountered: {final_state['errors']}")
    
    return final_state.get("final_response", "No response available"), final_state


if __name__ == "__main__":
    # Disable LangSmith for testing
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    tests = [
        "Find apartments in Mumbai under 50 lakh"
    ]
    
    for q in tests:
        print("\n" + "#" * 70)
        print(f"TEST QUERY: {q}")
        print("#" * 70)
        
        resp, st = run_agent_system(q)
        
        print("\n📋 RESPONSE:")
        print(resp)
        print("\n" + "="*70)
        print(f"Agents executed: {st.get('executed_agents', [])}")
        print(f"Agent call counts: {st.get('agent_call_count', {})}")
        if st.get('errors'):
            print(f"Errors: {st.get('errors')}")
        print()