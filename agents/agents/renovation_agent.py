"""
Renovation Estimation Agent
Uses LLM with few-shot examples for cost estimation
"""
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Optional

from config import GROQ_API_KEY, GROQ_MODELS, RENOVATION_COSTS
from state import AgentState


class RenovationAgent:
    """Estimates renovation costs based on property details"""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODELS["renovation"],
            temperature=0.2
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a renovation cost estimation expert for Indian real estate.

Given property details, estimate renovation costs based on:

Renovation Tiers (cost per sq ft in INR):
- Basic: ₹800-1,200 (paint, basic fixtures, minor repairs)
- Standard: ₹1,200-1,800 (flooring, electrical, plumbing, modular kitchen)
- Premium: ₹1,800-2,500 (false ceiling, designer fixtures, premium materials)
- Luxury: ₹2,500-4,000 (complete remodeling, high-end finishes, smart home)

Additional Costs:
- Balcony: +₹30,000-50,000
- Bathroom renovation: +₹80,000-150,000 per bathroom
- Kitchen: +₹100,000-300,000
- Living room false ceiling: +₹40,000-80,000

Return your estimate in this JSON format:
{{
  "tier": "standard",
  "base_cost": 850000,
  "additional_costs": {{
    "bathrooms": 160000,
    "kitchen": 150000
  }},
  "total_cost": 1160000,
  "breakdown": "Detailed explanation of costs",
  "assumptions": "List key assumptions made"
}}

Examples:

Property: 2BHK, 850 sq ft, 2 bathrooms, standard tier
Estimate:
{{
  "tier": "standard",
  "base_cost": 1275000,
  "additional_costs": {{"bathrooms": 160000}},
  "total_cost": 1435000,
  "breakdown": "850 sq ft × ₹1,500 = ₹12,75,000 base + 2 bathrooms × ₹80,000",
  "assumptions": "Standard tier, includes flooring, electrical, plumbing, paint"
}}"""),
            ("user", """Property Details:
{property_details}

Renovation Tier: {tier}

Estimate renovation cost:""")
        ])
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Estimate renovation costs
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with renovation estimate
        """
        try:
            # Get property details from SQL results or preferences
            property_details = self._extract_property_details(state)
            
            if not property_details:
                state["renovation_estimate"] = {
                    "error": "Insufficient property details for estimation"
                }
                state["executed_agents"].append("renovation_agent")
                return state
            
            # Determine tier from query or default to standard
            tier = self._determine_tier(state["user_query"])
            
            # Format prompt
            formatted_prompt = self.prompt.format_messages(
                property_details=str(property_details),
                tier=tier
            )
            
            # Get estimate
            response = self.llm.invoke(formatted_prompt)
            
            # Parse JSON response
            import json
            estimate = json.loads(response.content.strip())
            
            # Store results
            state["renovation_estimate"] = estimate
            state["executed_agents"].append("renovation_agent")
            
            print(f"✓ Renovation Agent: Estimated ₹{estimate.get('total_cost', 0):,} ({tier} tier)")
            
        except Exception as e:
            print(f"✗ Renovation Agent error: {e}")
            state["errors"].append(f"Renovation Agent: {str(e)}")
            state["renovation_estimate"] = {"error": str(e)}
        
        return state
    
    def _extract_property_details(self, state: AgentState) -> Optional[Dict]:
        """Extract property details from state"""
        details = {}
        
        # From SQL results
        if state.get("sql_results") and state["sql_results"].get("properties"):
            prop = state["sql_results"]["properties"][0]
            
            if prop.get("floorplan_parsed"):
                fp = prop["floorplan_parsed"]
                details["bedrooms"] = fp.get("rooms", 0)
                details["bathrooms"] = fp.get("bathrooms", 0)
                details["kitchens"] = fp.get("kitchens", 0)
                
                # Estimate area from room details
                if fp.get("rooms_detail"):
                    total_area = sum(room.get("approx_area", 0) for room in fp["rooms_detail"] if room.get("approx_area"))
                    if total_area > 0:
                        details["area_sqft"] = total_area
        
        # From preferences
        prefs = state.get("user_preferences", {})
        if "preferred_bedrooms" in prefs:
            details["bedrooms"] = prefs["preferred_bedrooms"]
        
        # Estimate area if not found
        if "area_sqft" not in details and "bedrooms" in details:
            # Rough estimate: 1BHK=500, 2BHK=800, 3BHK=1200
            area_map = {1: 500, 2: 800, 3: 1200, 4: 1500}
            details["area_sqft"] = area_map.get(details["bedrooms"], 1000)
        
        return details if details else None
    
    def _determine_tier(self, query: str) -> str:
        """Determine renovation tier from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["luxury", "premium", "high-end", "designer"]):
            return "luxury"
        elif any(word in query_lower for word in ["premium", "modern", "upgraded"]):
            return "premium"
        elif any(word in query_lower for word in ["basic", "simple", "minimal", "budget"]):
            return "basic"
        else:
            return "standard"


# Node function for LangGraph
def renovation_agent_node(state: AgentState) -> AgentState:
    """LangGraph node wrapper"""
    agent = RenovationAgent()
    return agent.execute(state)