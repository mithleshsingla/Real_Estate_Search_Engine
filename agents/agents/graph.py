"""
Fixed LangGraph multi-agent system graph file.

Changes made:
- Robust imports for langgraph checkpoint (works with both `langgraph.checkpoint` and `langgraph_checkpoint` package layouts).
- Safe fallback if checkpoint saver is missing (prints a clear warning and runs without persistent checkpointing).
- Improved logging and progress printing that doesn't assume the yielded state's internal structure.
- Small defensive changes to avoid exceptions when properties are missing from the state (e.g., executed_agents).
- Kept original logic and node wiring intact.

Usage:
- Place this file next to your other agent modules (router, planner, sql_agent, etc.)
- Ensure `state.AgentState` and other node functions are importable.
- Recommended: install `langgraph==0.2.16` and `langgraph-checkpoint==1.0.12` (or match imports below).

"""
from datetime import datetime
import uuid
import logging
from typing import Any, Dict, Tuple

# LangGraph imports - support multiple possible checkpoint package layouts
try:
    # Prefer the modern namespace if available
    from langgraph.checkpoint import SqliteSaver
except Exception:
    try:
        # Fallback to the older/alternate package namespace
        from langgraph_checkpoint.sqlite import SqliteSaver
    except Exception:
        SqliteSaver = None  # We'll handle missing saver gracefully below

from langgraph.graph import StateGraph, END
from langgraph.pregel import Channel, Pregel

# Your local project imports (must be available on PYTHONPATH / same package)
from state import AgentState, IntentType
from router import router_node
from planner import planner_node
from sql_agent import sql_agent_node
from rag_agent import rag_agent_node
from web_agent import web_agent_node
from renovation_agent import renovation_agent_node
from report_agent import report_agent_node, memory_node

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartsense.graph")


def should_continue_tasks(state: AgentState) -> str:
    """Determine next node based on current task list."""
    tasks = state.get("tasks", []) or []
    current_idx = state.get("current_task_index", 0) or 0

    # If no tasks or we've exhausted them, go to aggregator
    if current_idx >= len(tasks):
        return "aggregate"

    current_task = tasks[current_idx]
    # increment index in-state for next call
    state["current_task_index"] = current_idx + 1

    # Route to appropriate agent
    t = str(current_task).upper()
    if "SQL" in t:
        return "sql_agent"
    if "RAG" in t:
        return "rag_agent"
    if "WEB" in t:
        return "web_agent"
    if "RENOVATION" in t:
        return "renovation_agent"
    if "REPORT" in t:
        return "report_agent"

    return "aggregate"


def aggregate_results(state: AgentState) -> AgentState:
    """Aggregate partial results from various agents into final_response."""
    response_parts = []

    # SQL results
    sql_results = state.get("sql_results")
    if sql_results and isinstance(sql_results, dict) and sql_results.get("count", 0) > 0:
        count = sql_results.get("count", 0)
        response_parts.append(f"Found {count} properties matching your criteria.")

    # RAG results
    rag = state.get("rag_results")
    if rag and isinstance(rag, dict) and rag.get("answer"):
        response_parts.append(rag.get("answer"))

    # Web results
    web = state.get("web_results")
    if web and isinstance(web, dict) and web.get("insights"):
        response_parts.append(f"\n**Market Insights:**\n{web.get('insights')}")

    # Renovation
    ren = state.get("renovation_estimate")
    if ren and isinstance(ren, dict) and "total_cost" in ren:
        cost = ren.get("total_cost")
        response_parts.append(f"\n**Renovation Estimate:** ₹{cost:,}")

    # Report
    if state.get("report_path"):
        response_parts.append(f"\n**Report Generated:** {state.get('report_path')}")

    final_response = "\n\n".join(response_parts) if response_parts else ""

    # Add citations
    citations = state.get("citations") or []
    if citations:
        citations_text = "\n\n**References:**\n" + "\n".join([f"- {c}" for c in citations[:5]])
        final_response += citations_text

    state["final_response"] = final_response
    executed = state.get("executed_agents") or []
    if isinstance(executed, list):
        executed.append("aggregator")
    else:
        state["executed_agents"] = ["aggregator"]

    return state


def create_graph(checkpoint_conn: str = ":memory:") -> Any:
    """Create and compile the StateGraph workflow. Returns the compiled app.

    If SqliteSaver is available it will be used for checkpointer; otherwise a
    no-op (in-memory) checkpointer is used and a warning is logged.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("web_agent", web_agent_node)
    workflow.add_node("renovation_agent", renovation_agent_node)
    workflow.add_node("report_agent", report_agent_node)
    workflow.add_node("aggregate", aggregate_results)
    workflow.add_node("memory", memory_node)

    # Wiring
    workflow.set_entry_point("router")
    workflow.add_edge("router", "planner")

    workflow.add_conditional_edges(
        "planner",
        should_continue_tasks,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "web_agent": "web_agent",
            "renovation_agent": "renovation_agent",
            "report_agent": "report_agent",
            "aggregate": "aggregate",
        },
    )

    for agent in ["sql_agent", "rag_agent", "web_agent", "renovation_agent", "report_agent"]:
        workflow.add_conditional_edges(
            agent,
            should_continue_tasks,
            {
                "sql_agent": "sql_agent",
                "rag_agent": "rag_agent",
                "web_agent": "web_agent",
                "renovation_agent": "renovation_agent",
                "report_agent": "report_agent",
                "aggregate": "aggregate",
            },
        )

    workflow.add_edge("aggregate", "memory")
    workflow.add_edge("memory", END)

    # Checkpointer
    checkpointer = None
    if SqliteSaver is not None:
        try:
            # prefer connection-string or file API if present
            if hasattr(SqliteSaver, "from_conn_string"):
                checkpointer = SqliteSaver.from_conn_string(checkpoint_conn)
            elif hasattr(SqliteSaver, "from_file"):
                # fallback
                checkpointer = SqliteSaver.from_file(checkpoint_conn)
            else:
                # instantiate if a constructor exists
                checkpointer = SqliteSaver(checkpoint_conn) if callable(SqliteSaver) else None
        except Exception as e:
            logger.warning("Failed to initialize SqliteSaver: %s", e)
            checkpointer = None
    else:
        logger.warning("langgraph checkpoint saver not available; running without persistent checkpointing")

    app = workflow.compile(checkpointer=checkpointer)
    return app


def run_agent_system(query: str, session_id: str = None) -> Tuple[str, Dict[str, Any]]:
    """Run the multi-agent workflow and return final response + full state."""
    if not session_id:
        session_id = str(uuid.uuid4())

    initial_state = {
        "user_query": query,
        "session_id": session_id,
        "timestamp": datetime.now(),
        "intent": None,
        "tasks": [],
        "current_task_index": 0,
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
        "confidence_score": None,
    }

    app = create_graph()

    config = {"configurable": {"thread_id": session_id}}

    logger.info("RUNNING MULTI-AGENT SYSTEM | session=%s | query=%s", session_id, query)

    final_state = None
    # `app.stream` yields incremental states; be defensive when reading them
    for idx, state in enumerate(app.stream(initial_state, config)):
        final_state = state
        # Try to print a human-friendly progress line
        try:
            # many implementations yield a dict-like state where node name may be stored
            node_repr = getattr(state, "__repr__", None)
            print(f"[{idx}] progress... state received")
        except Exception:
            print(f"[{idx}] progress...")

    if final_state is None:
        logger.warning("No final state produced by the graph run")
        return "", {}

    executed_agents = final_state.get("executed_agents") or []
    errors = final_state.get("errors") or []

    logger.info("EXECUTION COMPLETE | agents=%s | errors=%s", executed_agents, errors)

    return final_state.get("final_response", ""), final_state


if __name__ == "__main__":
    test_queries = [
        "Find 3BHK apartments in Mumbai under 50 lakh",
        "Show luxury properties near tech parks with good schools",
        "I want to renovate a 2BHK apartment, estimate cost",
    ]

    for q in test_queries:
        print("\n" + "#" * 60)
        print("TEST:", q)
        print("#" * 60)
        resp, state = run_agent_system(q)
        print("\n📋 RESPONSE:\n", resp)
        print("\n")
