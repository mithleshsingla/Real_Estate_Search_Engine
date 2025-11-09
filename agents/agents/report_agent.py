"""
Report Generation Agent + Memory Component
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
import json
from pathlib import Path

from config import GOOGLE_API_KEY, GEMINI_MODELS, REPORT_TEMPLATE_DIR, REPORT_OUTPUT_DIR, POSTGRES_CONFIG
from state import AgentState
import psycopg2


# ============= REPORT AGENT =============
class ReportAgent:
    """Generates detailed PDF reports"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODELS["report"],
            google_api_key=GOOGLE_API_KEY,
            temperature=0.4
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """Generate HTML report (PDF conversion can be added later)"""
        try:
            # Gather all data
            sql_data = state.get("sql_results", {})
            rag_data = state.get("rag_results", {})
            web_data = state.get("web_results", {})
            renovation_data = state.get("renovation_estimate", {})
            
            # Generate HTML report
            html_content = self._generate_html_report(sql_data, rag_data, web_data, renovation_data)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = REPORT_OUTPUT_DIR / f"report_{timestamp}.html"
            report_path.write_text(html_content)
            
            state["report_path"] = str(report_path)
            state["executed_agents"].append("report_agent")
            
            print(f"✓ Report Agent: Generated report at {report_path}")
            
        except Exception as e:
            print(f"✗ Report Agent error: {e}")
            state["errors"].append(f"Report Agent: {str(e)}")
        
        return state
    
    def _generate_html_report(self, sql_data, rag_data, web_data, renovation_data):
        """Generate simple HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Property Report</title>
    <style>
        body {{ font-family: Arial; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
        table {{ width: 100%; border-collapse: collapse; }}
        td, th {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
    </style>
</head>
<body>
    <h1>Property Search Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    
    <div class="section">
        <h2>Search Results</h2>
        <p>Found {sql_data.get('count', 0)} properties</p>
        {self._format_properties_table(sql_data.get('properties', []))}
    </div>
    
    <div class="section">
        <h2>Market Insights</h2>
        <p>{web_data.get('insights', 'No market data available')}</p>
    </div>
    
    <div class="section">
        <h2>Renovation Estimate</h2>
        <p>Total Cost: ₹{renovation_data.get('total_cost', 0):,}</p>
        <p>{renovation_data.get('breakdown', '')}</p>
    </div>
</body>
</html>
"""
        return html
    
    def _format_properties_table(self, properties):
        if not properties:
            return "<p>No properties found</p>"
        
        rows = ""
        for prop in properties[:5]:
            rows += f"""
            <tr>
                <td>{prop.get('property_id', '')}</td>
                <td>{prop.get('title', '')}</td>
                <td>{prop.get('location', '')}</td>
                <td>₹{prop.get('price', 0):,}</td>
            </tr>
            """
        
        return f"""
        <table>
            <tr><th>ID</th><th>Title</th><th>Location</th><th>Price</th></tr>
            {rows}
        </table>
        """


def report_agent_node(state: AgentState) -> AgentState:
    agent = ReportAgent()
    return agent.execute(state)


# ============= MEMORY COMPONENT =============
class MemoryManager:
    """Manages user preferences and conversation history"""
    
    def __init__(self):
        self.db_config = POSTGRES_CONFIG
        self._ensure_memory_tables()
    
    def _ensure_memory_tables(self):
        """Create memory tables if not exist"""
        schema = """
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id VARCHAR(255) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT NOW(),
            last_active TIMESTAMP DEFAULT NOW(),
            preferences JSONB
        );
        
        CREATE TABLE IF NOT EXISTS conversation_history (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255),
            timestamp TIMESTAMP DEFAULT NOW(),
            role VARCHAR(20),
            content TEXT,
            FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
        );
        """
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(schema)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠ Memory table creation: {e}")
    
    def save_conversation(self, session_id: str, role: str, content: str):
        """Save message to history"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Ensure session exists first
            cursor.execute(
                """INSERT INTO user_sessions (session_id, preferences) 
                   VALUES (%s, '{}') 
                   ON CONFLICT (session_id) DO UPDATE SET last_active = NOW()""",
                (session_id,)
            )
            
            # Then insert conversation
            cursor.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠ Save conversation error: {e}")
            # Don't raise - memory errors shouldn't crash the system
    
    def get_conversation_history(self, session_id: str, limit: int = 10):
        """Get recent conversation"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM conversation_history WHERE session_id = %s ORDER BY timestamp DESC LIMIT %s",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            conn.close()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
        except:
            return []
    
    def save_preferences(self, session_id: str, preferences: dict):
        """Save user preferences"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO user_sessions (session_id, preferences) VALUES (%s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET preferences = %s, last_active = NOW()""",
                (session_id, json.dumps(preferences), json.dumps(preferences))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠ Save preferences error: {e}")
    
    def get_preferences(self, session_id: str):
        """Get user preferences"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT preferences FROM user_sessions WHERE session_id = %s", (session_id,))
            row = cursor.fetchone()
            conn.close()
            return json.loads(row[0]) if row and row[0] else {}
        except:
            return {}


def memory_node(state: AgentState) -> AgentState:
    """Update memory with conversation"""
    try:
        memory = MemoryManager()
        session_id = state.get("session_id", "default")
        
        # Save user query
        memory.save_conversation(session_id, "user", state["user_query"])
        
        # Save response if available
        if state.get("final_response"):
            memory.save_conversation(session_id, "assistant", state["final_response"])
        
        # Save preferences
        if state.get("user_preferences"):
            memory.save_preferences(session_id, state["user_preferences"])
        
        # Load history for context
        state["conversation_history"] = memory.get_conversation_history(session_id)
        
        print(f"✓ Memory: Saved session {session_id}")
        
    except Exception as e:
        print(f"⚠ Memory error: {e}")
    
    return state