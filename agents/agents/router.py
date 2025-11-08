"""
Query Router Agent - Intent Detection & Slot Extraction
Uses Groq for fast classification
"""
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import json

from config import GROQ_API_KEY, GROQ_MODELS, ROUTER_TEMPERATURE
from state import AgentState, IntentType


class RouterOutput(BaseModel):
    """Structured output from router"""
    intent: str = Field(description="Classified intent type")
    confidence: float = Field(description="Confidence score 0-1")
    extracted_slots: dict = Field(description="Extracted entities (location, bedrooms, budget, etc.)")
    reasoning: str = Field(description="Brief explanation of classification")


class QueryRouter:
    """Routes queries to appropriate agents based on intent"""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODELS["router"],
            temperature=ROUTER_TEMPERATURE
        )
        self.parser = PydanticOutputParser(pydantic_object=RouterOutput)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query router for a real estate property search system.

Classify user queries into one of these intents:
1. SIMPLE_SEARCH - Direct property search with filters (location, price, bedrooms)
   Examples: "3BHK in Mumbai under 50 lakh", "apartments in Bangalore"

2. SEMANTIC_SEARCH - Complex/fuzzy search requiring RAG
   Examples: "properties near tech parks", "family-friendly neighborhoods", "luxury homes with pool"

3. ESTIMATION - Renovation cost estimation
   Examples: "renovation cost for 2BHK", "how much to renovate apartment"

4. REPORT_GENERATION - Generate detailed property report
   Examples: "generate report for PROP_123", "detailed analysis of property"

5. COMPLEX_QUERY - Multi-step query requiring web research
   Examples: "find 2BHK in Pune, check market rates, estimate renovation"

6. MEMORY_QUERY - Access saved preferences or properties
   Examples: "show my saved properties", "what's my budget", "remember I like Mumbai"

Extract these slots if present:
- location: city/area name
- bedrooms: number (1,2,3,4+)
- budget_min/budget_max: price range in lakhs
- property_type: apartment/villa/penthouse
- amenities: list of desired features
- property_id: specific property ID mentioned

{format_instructions}"""),
            ("user", "{query}")
        ])
    
    def route(self, state: AgentState) -> AgentState:
        """
        Classify intent and extract slots from user query
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with intent and slots
        """
        query = state["user_query"]
        
        try:
            # Format prompt
            formatted_prompt = self.prompt.format_messages(
                query=query,
                format_instructions=self.parser.get_format_instructions()
            )
            
            # Get response
            response = self.llm.invoke(formatted_prompt)
            
            # Parse structured output
            result = self.parser.parse(response.content)
            
            # Update state
            state["intent"] = result.intent
            state["confidence_score"] = result.confidence
            state["executed_agents"].append("router")
            
            # Store extracted slots in user preferences for context
            if "user_preferences" not in state:
                state["user_preferences"] = {}
            
            slots = result.extracted_slots
            
            # Update preferences with extracted slots
            if "location" in slots and slots["location"]:
                if "preferred_locations" not in state["user_preferences"]:
                    state["user_preferences"]["preferred_locations"] = []
                state["user_preferences"]["preferred_locations"].append(slots["location"])
            
            if "bedrooms" in slots and slots["bedrooms"]:
                state["user_preferences"]["preferred_bedrooms"] = slots["bedrooms"]
            
            if "budget_min" in slots and slots["budget_min"]:
                state["user_preferences"]["budget_min"] = slots["budget_min"]
            
            if "budget_max" in slots and slots["budget_max"]:
                state["user_preferences"]["budget_max"] = slots["budget_max"]
            
            print(f"✓ Router: Intent={result.intent}, Confidence={result.confidence:.2f}")
            print(f"  Slots: {slots}")
            
        except Exception as e:
            print(f"✗ Router error: {e}")
            state["errors"].append(f"Router: {str(e)}")
            # Default to semantic search on error
            state["intent"] = IntentType.SEMANTIC_SEARCH
            state["confidence_score"] = 0.5
        
        return state


# Node function for LangGraph
def router_node(state: AgentState) -> AgentState:
    """LangGraph node wrapper"""
    router = QueryRouter()
    return router.route(state)