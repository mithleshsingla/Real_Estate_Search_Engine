"""
State definition for LangGraph multi-agent system
WITH AGENT EXECUTION TRACKING TO PREVENT LOOPS
"""
from typing import TypedDict, List, Dict, Optional, Annotated
from datetime import datetime
import operator


class AgentState(TypedDict):
    """Shared state across all agents"""
    
    # Input
    user_query: str
    session_id: str
    timestamp: datetime
    
    # Routing & Planning
    intent: Optional[str]  # search, estimate, report, complex
    tasks: Annotated[List[str], operator.add]  # List of tasks to execute
    current_task_index: int
    
    # NEW: Agent execution tracking to prevent loops
    agent_call_count: Dict[str, int]  # Track how many times each agent was called
    max_agent_retries: int  # Maximum retries per agent (default: 3)
    task_completed: bool  # Flag to indicate current task is done
    
    # Agent Results
    sql_results: Optional[Dict]  # Results from structured queries
    rag_results: Optional[Dict]  # Results from vector search
    web_results: Optional[Dict]  # External web research data
    renovation_estimate: Optional[Dict]  # Renovation cost breakdown
    report_path: Optional[str]  # Path to generated PDF report
    
    # Response Generation
    intermediate_responses: Annotated[List[str], operator.add]  # Partial responses
    final_response: str  # Final answer to user
    citations: Annotated[List[str], operator.add]  # Academic citations
    
    # Memory & Context
    conversation_history: List[Dict]  # Previous messages in session
    user_preferences: Dict  # Stored user preferences
    
    # Metadata
    executed_agents: Annotated[List[str], operator.add]  # Track which agents ran
    errors: Annotated[List[str], operator.add]  # Track any errors
    confidence_score: Optional[float]  # Overall confidence in response


class UserPreferences(TypedDict):
    """User preferences stored in memory"""
    preferred_locations: List[str]
    budget_min: Optional[int]
    budget_max: Optional[int]
    preferred_bedrooms: Optional[int]
    preferred_amenities: List[str]
    saved_properties: List[str]  # Property IDs


class TaskType:
    """Task types for planner"""
    SEARCH = "search"
    ESTIMATE = "estimate"
    REPORT = "report"
    RESEARCH = "research"
    AGGREGATE = "aggregate"


class IntentType:
    """Intent classification"""
    SIMPLE_SEARCH = "simple_search"  # Direct SQL query
    SEMANTIC_SEARCH = "semantic_search"  # RAG needed
    ESTIMATION = "estimation"  # Renovation cost
    REPORT_GENERATION = "report_generation"  # PDF report
    COMPLEX_QUERY = "complex_query"  # Multi-step with web research
    MEMORY_QUERY = "memory_query"  # Access saved preferences