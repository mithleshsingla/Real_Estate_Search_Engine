"""
Planner Agent - Task Decomposition
Uses Gemini for complex reasoning
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

from config import GOOGLE_API_KEY, GEMINI_MODELS, PLANNER_TEMPERATURE
from state import AgentState, IntentType, TaskType


class PlannerOutput(BaseModel):
    """Structured task plan"""
    tasks: List[str] = Field(description="Ordered list of tasks to execute")
    reasoning: str = Field(description="Explanation of the plan")
    estimated_time: str = Field(description="Estimated execution time")


class TaskPlanner:
    """Decomposes complex queries into ordered tasks"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODELS["planner"],
            google_api_key=GOOGLE_API_KEY,
            temperature=PLANNER_TEMPERATURE
        )
        self.parser = PydanticOutputParser(pydantic_object=PlannerOutput)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a task planning agent for a real estate search system.

Given a user query and its classified intent, break it down into ordered tasks.

Available task types:
- SQL_SEARCH: Query structured database for properties
- RAG_SEARCH: Semantic search in vector database
- WEB_RESEARCH: Fetch external market data via web search
- RENOVATION_ESTIMATE: Calculate renovation costs
- REPORT_GENERATE: Create PDF report
- AGGREGATE: Combine results from multiple sources

Task planning rules:
1. SQL_SEARCH and RAG_SEARCH can run in parallel
2. WEB_RESEARCH can run parallel with searches
3. RENOVATION_ESTIMATE needs property data first
4. REPORT_GENERATE must be last
5. AGGREGATE combines parallel task results

Examples:

Query: "Find 2BHK in Mumbai under 50 lakh"
Intent: SIMPLE_SEARCH
Tasks: ["SQL_SEARCH"]

Query: "Show luxury properties near IT parks with good schools"
Intent: SEMANTIC_SEARCH
Tasks: ["SQL_SEARCH", "RAG_SEARCH", "WEB_RESEARCH", "AGGREGATE"]

Query: "Find 3BHK in Pune, check market rates, estimate renovation, create report"
Intent: COMPLEX_QUERY
Tasks: ["SQL_SEARCH", "WEB_RESEARCH", "RENOVATION_ESTIMATE", "AGGREGATE", "REPORT_GENERATE"]

{format_instructions}"""),
            ("user", """Query: {query}
Intent: {intent}
User Preferences: {preferences}

Generate task plan:""")
        ])
    
    def plan(self, state: AgentState) -> AgentState:
        """
        Create task execution plan
        
        Args:
            state: Current agent state with intent
            
        Returns:
            Updated state with task list
        """
        try:
            # Get query and intent
            query = state["user_query"]
            intent = state.get("intent", IntentType.SEMANTIC_SEARCH)
            preferences = state.get("user_preferences", {})
            
            # Fast path for simple intents
            if intent == IntentType.SIMPLE_SEARCH:
                state["tasks"] = ["SQL_SEARCH"]
                state["executed_agents"].append("planner")
                print(f"✓ Planner: Fast path - SQL_SEARCH only")
                return state
            
            # Format prompt
            formatted_prompt = self.prompt.format_messages(
                query=query,
                intent=intent,
                preferences=str(preferences),
                format_instructions=self.parser.get_format_instructions()
            )
            
            # Get plan
            response = self.llm.invoke(formatted_prompt)
            result = self.parser.parse(response.content)
            
            # Update state
            state["tasks"] = result.tasks
            state["current_task_index"] = 0
            state["executed_agents"].append("planner")
            
            print(f"✓ Planner: Generated {len(result.tasks)} tasks")
            print(f"  Tasks: {result.tasks}")
            print(f"  Reasoning: {result.reasoning}")
            
        except Exception as e:
            print(f"✗ Planner error: {e}")
            state["errors"].append(f"Planner: {str(e)}")
            # Fallback plan
            state["tasks"] = ["RAG_SEARCH"]
            state["current_task_index"] = 0
        
        return state


# Node function for LangGraph
def planner_node(state: AgentState) -> AgentState:
    """LangGraph node wrapper"""
    planner = TaskPlanner()
    return planner.plan(state)