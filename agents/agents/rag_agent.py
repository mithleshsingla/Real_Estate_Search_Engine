"""
RAG Agent - Semantic Search with Vector Database
Uses Gemini for synthesis and citation generation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "etl"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from vector_store import VectorStore

from config import GOOGLE_API_KEY, GEMINI_MODELS, RAG_TEMPERATURE
from state import AgentState


class RAGAgent:
    """Retrieves and synthesizes information from vector database"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODELS["rag"],
            google_api_key=GOOGLE_API_KEY,
            temperature=RAG_TEMPERATURE
        )
        self.vector_store = VectorStore()
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a real estate information synthesis expert.

Given a user query and retrieved property documents, synthesize a comprehensive answer.

Rules:
1. Use information from the retrieved documents
2. Be specific and detailed
3. Mention property IDs when referencing specific properties
4. If documents don't fully answer the query, acknowledge limitations
5. Maintain professional, helpful tone

Format your response as:
ANSWER: [Your synthesized answer]
RELEVANT_IDS: [comma-separated property IDs mentioned]"""),
            ("user", """Query: {query}

Retrieved Documents:
{documents}

Synthesize answer:""")
        ])
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute semantic search and synthesis
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with RAG results
        """
        try:
            query = state["user_query"]
            
            # Vector search
            search_results = self.vector_store.search_similar(query, limit=5)
            
            print(f"✓ RAG Agent: Found {len(search_results)} similar properties")
            
            if not search_results:
                state["rag_results"] = {
                    "answer": "No relevant properties found in vector search.",
                    "sources": []
                }
                state["executed_agents"].append("rag_agent")
                return state
            
            # Format documents for synthesis
            docs_text = self._format_documents(search_results)
            
            # Synthesize answer
            formatted_prompt = self.synthesis_prompt.format_messages(
                query=query,
                documents=docs_text
            )
            
            response = self.llm.invoke(formatted_prompt)
            answer_text = response.content
            
            # Parse response
            if "ANSWER:" in answer_text:
                answer = answer_text.split("ANSWER:")[1].split("RELEVANT_IDS:")[0].strip()
                relevant_ids = answer_text.split("RELEVANT_IDS:")[1].strip() if "RELEVANT_IDS:" in answer_text else ""
            else:
                answer = answer_text
                relevant_ids = ""
            
            # Store results
            state["rag_results"] = {
                "answer": answer,
                "sources": search_results,
                "relevant_ids": [id.strip() for id in relevant_ids.split(",") if id.strip()]
            }
            
            state["executed_agents"].append("rag_agent")
            
            # Generate citations
            for result in search_results:
                citation = self._format_citation(result)
                state["citations"].append(citation)
            
            print(f"✓ RAG Agent: Synthesized answer ({len(answer)} chars)")
            
        except Exception as e:
            print(f"✗ RAG Agent error: {e}")
            state["errors"].append(f"RAG Agent: {str(e)}")
            state["rag_results"] = {"answer": "Error in semantic search", "sources": []}
        
        return state
    
    def _format_documents(self, results: list) -> str:
        """Format search results as document context"""
        docs = []
        for i, result in enumerate(results, 1):
            doc = f"""
Document {i}:
Property ID: {result['property_id']}
Title: {result['title']}
Location: {result['location']}
Price: ₹{result['price']:,}
Similarity Score: {result['score']:.3f}
"""
            docs.append(doc)
        return "\n".join(docs)
    
    def _format_citation(self, result: dict) -> str:
        """Format academic citation from search result"""
        title = result.get("title", "Unknown Property")
        location = result.get("location", "Unknown")
        prop_id = result.get("property_id", "Unknown")
        
        return f"{title} in {location} ({prop_id}, 2024)"


# Node function for LangGraph
def rag_agent_node(state: AgentState) -> AgentState:
    """LangGraph node wrapper"""
    agent = RAGAgent()
    return agent.execute(state)