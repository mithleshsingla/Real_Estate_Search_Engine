"""
Structured Data Agent - SQL Query Generation & Execution
Uses Groq for fast SQL generation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "tools"))

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict

from config import GROQ_API_KEY, GROQ_MODELS
from state import AgentState
from sql_tools import SQLTools


class StructuredDataAgent:
    """Executes structured queries against PostgreSQL"""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODELS["sql"],
            temperature=0.1
        )
        self.tools = SQLTools()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL query parameter extractor for a real estate database.

Given a user query and preferences, extract the search parameters.

Available filters:
- location: string (city or area)
- min_price: integer (in lakhs, multiply by 100000 for INR)
- max_price: integer (in lakhs, multiply by 100000 for INR)
- bedrooms: integer (1, 2, 3, 4+)
- property_type: string (apartment, villa, penthouse)

Return ONLY a JSON object with extracted parameters. Example:
{{"location": "Mumbai", "min_price": 3000000, "max_price": 5000000, "bedrooms": 3}}

If no filters are found, return: {{"limit": 10}}"""),
            ("user", """Query: {query}
Preferences: {preferences}

Extract search parameters:""")
        ])
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute SQL search
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with SQL results
        """
        try:
            query = state["user_query"]
            preferences = state.get("user_preferences", {})
            
            # Extract search parameters using LLM
            formatted_prompt = self.prompt.format_messages(
                query=query,
                preferences=str(preferences)
            )
            
            response = self.llm.invoke(formatted_prompt)
            
            # Parse parameters
            import json
            params = json.loads(response.content.strip())
            
            print(f"✓ SQL Agent: Extracted params: {params}")
            
            # Execute search
            results = self.tools.search_properties(**params)
            
            # Store results
            state["sql_results"] = {
                "count": len(results),
                "properties": results[:10],  # Limit to 10 for token efficiency
                "query_params": params
            }
            
            state["executed_agents"].append("sql_agent")
            
            print(f"✓ SQL Agent: Found {len(results)} properties")
            
            # Generate citations
            for prop in results[:5]:  # Cite top 5
                citation = self._format_citation(prop)
                state["citations"].append(citation)
            
        except Exception as e:
            print(f"✗ SQL Agent error: {e}")
            state["errors"].append(f"SQL Agent: {str(e)}")
            state["sql_results"] = {"count": 0, "properties": []}
        
        return state
    
    def _format_citation(self, property_data: Dict) -> str:
        """Format academic citation"""
        title = property_data.get("title", "Unknown")
        location = property_data.get("location", "Unknown")
        prop_id = property_data.get("property_id", "Unknown")
        year = property_data.get("listing_date", "").split("-")[0] if property_data.get("listing_date") else "2024"
        
        return f"{title} in {location} ({prop_id}, {year})"


# Node function for LangGraph
def sql_agent_node(state: AgentState) -> AgentState:
    """LangGraph node wrapper"""
    agent = StructuredDataAgent()
    return agent.execute(state)