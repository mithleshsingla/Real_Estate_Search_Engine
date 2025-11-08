"""
FastAPI Backend for Multi-Agent Real Estate System
"""
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent / "agents"))
sys.path.append(str(Path(__file__).parent.parent / "etl"))
sys.path.append(str(Path(__file__).parent.parent / "floorplan_parser"))

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import uuid
from datetime import datetime
import json

# Import ETL pipeline
from etl_pipeline import ETLPipeline

# Import floorplan parser
from floorplan_inference import FloorplanParser

# Import agent graph
from graph import run_agent_system

# Import config
from config import CHECKPOINT_PATH, ROOM_CLASSIFIER_PATH


app = FastAPI(
    title="Real Estate Multi-Agent System",
    description="AI-powered property search with ETL, floorplan parsing, and multi-agent chat",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    executed_agents: List[str]
    citations: List[str]
    errors: List[str]


class IngestionRequest(BaseModel):
    excel_path: Optional[str] = None
    recreate_collections: bool = False


# Global state
floorplan_parser_instance = None
active_websockets = {}


@app.get("/")
async def root():
    return {
        "message": "Real Estate Multi-Agent System API",
        "version": "2.0.0",
        "endpoints": {
            "POST /ingest": "Trigger ETL pipeline",
            "POST /parse-floorplan": "Parse single floorplan image",
            "POST /chat": "Chat with multi-agent system",
            "WS /ws/chat": "WebSocket chat endpoint",
            "GET /reports/{filename}": "Download generated reports"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    try:
        # Test DB connections
        from database import DatabaseHandler
        db = DatabaseHandler()
        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "postgres": "connected",
            "agents": "ready"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ============= ETL ENDPOINTS =============
@app.post("/ingest")
async def ingest_data(request: IngestionRequest, background_tasks: BackgroundTasks):
    """Trigger ETL pipeline"""
    def run_etl():
        try:
            pipeline = ETLPipeline()
            excel_path = Path(request.excel_path) if request.excel_path else None
            result = pipeline.run_ingestion(
                excel_path=excel_path,
                recreate_collections=request.recreate_collections
            )
            print(f"✓ ETL completed: {result.successful}/{result.total_properties} successful")
        except Exception as e:
            print(f"✗ ETL failed: {e}")
    
    background_tasks.add_task(run_etl)
    
    return {
        "status": "started",
        "message": "ETL pipeline started in background"
    }


@app.post("/parse-floorplan")
async def parse_floorplan(file: UploadFile = File(...)):
    """Debug endpoint to parse single floorplan"""
    global floorplan_parser_instance
    
    try:
        # Initialize parser (lazy)
        if floorplan_parser_instance is None:
            from floorplan_inference import FloorplanParser
            floorplan_parser_instance = FloorplanParser(
                checkpoint_path=str(CHECKPOINT_PATH),
                ml_classifier_path=str(ROOM_CLASSIFIER_PATH)
            )
        
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Parse
        output, detections = floorplan_parser_instance.parse(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        
        return {
            "filename": file.filename,
            "parsed_data": output,
            "detections_count": len(detections)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


# ============= CHAT ENDPOINTS =============
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """REST endpoint for chat"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Run agent system
        response, state = run_agent_system(request.query, session_id)
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            executed_agents=state.get("executed_agents", []),
            citations=state.get("citations", []),
            errors=state.get("errors", [])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    active_websockets[session_id] = websocket
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            query = json.loads(data)["query"]
            
            # Send typing indicator
            await websocket.send_json({"type": "typing", "message": "Agent system processing..."})
            
            # Run agents
            response, state = run_agent_system(query, session_id)
            
            # Send response
            await websocket.send_json({
                "type": "response",
                "message": response,
                "executed_agents": state.get("executed_agents", []),
                "citations": state.get("citations", [])
            })
    
    except WebSocketDisconnect:
        del active_websockets[session_id]
        print(f"WebSocket disconnected: {session_id}")


@app.get("/reports/{filename}")
async def download_report(filename: str):
    """Download generated report"""
    from config import REPORT_OUTPUT_DIR
    report_path = REPORT_OUTPUT_DIR / filename
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(report_path, media_type="text/html", filename=filename)


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    try:
        from report_agent import MemoryManager
        memory = MemoryManager()
        history = memory.get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)