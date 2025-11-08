"""
Web Research Agent - External Data via Tavily
Uses Groq for fast synthesis
"""
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import ChatPromptTemplate

from config import GROQ_API_KEY, GROQ_MODELS, TAVILY_API_KEY, TAVILY_MAX_RESULTS, TAVILY_SEARCH_DEPTH
from state import AgentState


class WebResearchAgent:
    """Fetches external market data and neighborhood information"""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODELS["web"],
            temperature=0.3
        )
        
        self.search_tool = TavilySearchResults(
            api_key=TAVILY_API_KEY,
            max_results=TAVILY_MAX_RESULTS,
            search_depth=TAVILY_SEARCH_DEPTH,
            include_answer=True,
            include_raw_content=False
        )
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a real estate market research analyst.

Given web search results about a location, synthesize key insights about:
1. Current market rates (price per sq ft)
2. Neighborhood information (schools, hospitals, transport)
3. Recent developments or trends
4. Amenities and infrastructure

Be concise and factual. Cite sources when possible."""),
            ("user", """Location: {location}
Query Context: {query}

Search Results:
{results}

Synthesize market insights:""")
        ])
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute web research
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with web research results
        """
        try:
            query = state["user_query"]
            preferences = state.get("user_preferences", {})
            
            # Extract location from preferences or query
            location = None
            if "preferred_locations" in preferences and preferences["preferred_locations"]:
                location = preferences["preferred_locations"][-1]
            
            if not location:
                # Try to extract from query
                location = self._extract_location(query)
            
            if not location:
                state["web_results"] = {
                    "message": "No location specified for web research",
                    "insights": []
                }
                state["executed_agents"].append("web_agent")
                return state
            
            # Build search queries
            search_queries = [
                f"real estate market rates {location} India 2024",
                f"property prices per square foot {location}",
                f"neighborhood amenities {location} schools hospitals",
                f"infrastructure development {location} 2024"
            ]
            
            # Execute searches
            all_results = []
            for search_query in search_queries:
                try:
                    results = self.search_tool.invoke({"query": search_query})
                    if results:
                        all_results.extend(results if isinstance(results, list) else [results])
                except Exception as e:
                    print(f"⚠ Tavily search failed for '{search_query}': {e}")
            
            print(f"✓ Web Agent: Retrieved {len(all_results)} search results for {location}")
            
            if not all_results:
                state["web_results"] = {
                    "location": location,
                    "message": "No web results found",
                    "insights": []
                }
                state["executed_agents"].append("web_agent")
                return state
            
            # Synthesize insights
            results_text = self._format_results(all_results)
            
            formatted_prompt = self.synthesis_prompt.format_messages(
                location=location,
                query=query,
                results=results_text
            )
            
            response = self.llm.invoke(formatted_prompt)
            insights = response.content
            
            # Store results
            state["web_results"] = {
                "location": location,
                "insights": insights,
                "raw_results": all_results[:5],  # Keep top 5
                "sources": [r.get("url", "") for r in all_results[:5] if isinstance(r, dict)]
            }
            
            state["executed_agents"].append("web_agent")
            
            # Add citations
            state["citations"].append(f"Market Research on {location} (Tavily Search, 2024)")
            
            print(f"✓ Web Agent: Synthesized insights for {location}")
            
        except Exception as e:
            print(f"✗ Web Agent error: {e}")
            state["errors"].append(f"Web Agent: {str(e)}")
            state["web_results"] = {"message": "Error in web research", "insights": []}
        
        return state
    
    def _extract_location(self, query: str) -> str:
        """Simple location extraction"""
        # Common Indian cities
        cities = ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad", "Chennai", 
                  "Kolkata", "Ahmedabad", "Gurgaon", "Noida"]
        
        query_lower = query.lower()
        for city in cities:
            if city.lower() in query_lower:
                return city
        
        return None
    
    def _format_results(self, results: list) -> str:
        """Format search results"""
        formatted = []
        for i, result in enumerate(results[:10], 1):  # Top 10
            if isinstance(result, dict):
                content = result.get("content", "")
                url = result.get("url", "")
                formatted.append(f"{i}. {content}\nSource: {url}\n")
        
        return "\n".join(formatted)


# Node function for LangGraph
def web_agent_node(state: AgentState) -> AgentState:
    """LangGraph node wrapper"""
    agent = WebResearchAgent()
    return agent.execute(state)