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
import traceback
import tempfile
import shutil

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
            "POST /ingest-file": "Upload and trigger ETL pipeline (NEW)",
            "POST /ingest": "Trigger ETL with file path",
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
@app.post("/ingest-file")
async def ingest_file(
    file: UploadFile = File(...),
    recreate_collections: str = "false",
    background_tasks: BackgroundTasks = None
):
    """
    Upload Excel file and trigger ETL pipeline
    This is the recommended endpoint for file uploads from Streamlit
    """
    
    try:
        # Create persistent temp directory
        temp_dir = Path(tempfile.gettempdir()) / "smartsense_uploads"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file_path = temp_dir / f"{timestamp}_{file.filename}"
        
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        file_size = temp_file_path.stat().st_size / 1024
        print(f"\n📁 File uploaded: {file.filename}")
        print(f"💾 Saved to: {temp_file_path}")
        print(f"📊 Size: {file_size:.2f} KB")
        
        # Convert string to boolean
        recreate = recreate_collections.lower() == "true"
        
        # Background task for ETL processing
        def run_etl():
            try:
                print("\n" + "="*80)
                print("🚀 STARTING ETL PIPELINE")
                print("="*80)
                print(f"📄 File: {temp_file_path}")
                print(f"🔄 Recreate: {recreate}")
                
                pipeline = ETLPipeline()
                result = pipeline.run_ingestion(
                    excel_path=temp_file_path,
                    recreate_collections=recreate
                )
                
                print("\n" + "="*80)
                print("✅ ETL PIPELINE COMPLETED")
                print("="*80)
                print(f"📊 Total: {result.total_properties}")
                print(f"✓ Successful: {result.successful}")
                print(f"✗ Failed: {result.failed}")
                print(f"⏱️ Time: {result.processing_time:.2f}s")
                print("="*80 + "\n")
                
                # Clean up temp file after successful processing
                try:
                    temp_file_path.unlink()
                    print(f"🗑️ Cleaned up: {temp_file_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ Could not delete temp file: {cleanup_error}")
                
            except Exception as e:
                print("\n" + "="*80)
                print("❌ ETL PIPELINE FAILED")
                print("="*80)
                print(f"Error: {e}")
                traceback.print_exc()
                print("="*80 + "\n")
                
                # Clean up temp file on error
                try:
                    if temp_file_path.exists():
                        temp_file_path.unlink()
                        print(f"🗑️ Cleaned up temp file after error")
                except:
                    pass
        
        # Add background task
        background_tasks.add_task(run_etl)
        
        return {
            "status": "started",
            "message": "File uploaded successfully. ETL pipeline started in background.",
            "filename": file.filename,
            "file_size_kb": round(file_size, 2),
            "temp_path": str(temp_file_path),
            "recreate_collections": recreate
        }
        
    except Exception as e:
        print(f"❌ File upload failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.post("/ingest")
async def ingest_data(request: IngestionRequest, background_tasks: BackgroundTasks):
    """
    Trigger ETL pipeline with existing file path
    Use /ingest-file for uploading files from Streamlit
    """
    
    # Validate file path if provided
    if request.excel_path:
        excel_path = Path(request.excel_path)
        if not excel_path.exists():
            raise HTTPException(
                status_code=400, 
                detail=f"Excel file not found: {request.excel_path}"
            )
        print(f"📁 Using Excel file: {excel_path}")
    else:
        print("📁 Using default Excel file from config")
    
    def run_etl():
        try:
            print("\n" + "="*80)
            print("🚀 STARTING ETL PIPELINE")
            print("="*80)
            
            pipeline = ETLPipeline()
            excel_path = Path(request.excel_path) if request.excel_path else None
            result = pipeline.run_ingestion(
                excel_path=excel_path,
                recreate_collections=request.recreate_collections
            )
            
            print("\n" + "="*80)
            print("✅ ETL PIPELINE COMPLETED")
            print("="*80)
            print(f"📊 Total: {result.total_properties}")
            print(f"✓ Successful: {result.successful}")
            print(f"✗ Failed: {result.failed}")
            print(f"⏱️ Time: {result.processing_time:.2f}s")
            print("="*80 + "\n")
            
        except Exception as e:
            print("\n" + "="*80)
            print("❌ ETL PIPELINE FAILED")
            print("="*80)
            print(f"Error: {e}")
            traceback.print_exc()
            print("="*80 + "\n")
    
    background_tasks.add_task(run_etl)
    
    return {
        "status": "started",
        "message": "ETL pipeline started in background",
        "excel_path": request.excel_path,
        "recreate_collections": request.recreate_collections
    }


@app.post("/parse-floorplan")
async def parse_floorplan(file: UploadFile = File(...)):
    """Parse single floorplan image and return structured data"""
    global floorplan_parser_instance
    
    try:
        # Initialize parser (lazy loading)
        if floorplan_parser_instance is None:
            print("🔧 Initializing floorplan parser...")
            from floorplan_inference import FloorplanParser
            floorplan_parser_instance = FloorplanParser(
                checkpoint_path=str(CHECKPOINT_PATH),
                ml_classifier_path=str(ROOM_CLASSIFIER_PATH)
            )
            print("✓ Floorplan parser initialized")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = Path(tmp_file.name)
        
        print(f"📐 Parsing floorplan: {file.filename}")
        
        # Parse floorplan
        output, detections = floorplan_parser_instance.parse(str(temp_path))
        
        # Clean up temp file
        temp_path.unlink()
        
        print(f"✓ Parsed successfully: {len(detections)} detections")
        
        return {
            "filename": file.filename,
            "parsed_data": output,
            "detections_count": len(detections)
        }
    
    except Exception as e:
        print(f"❌ Floorplan parsing failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


# ============= CHAT ENDPOINTS =============
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """REST endpoint for chat with multi-agent system"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        print(f"\n💬 Chat request: {request.query[:50]}...")
        print(f"📝 Session: {session_id}")
        
        # Run agent system
        response, state = run_agent_system(request.query, session_id)
        
        print(f"✓ Response generated")
        print(f"🤖 Agents: {', '.join(state.get('executed_agents', []))}")
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            executed_agents=state.get("executed_agents", []),
            citations=state.get("citations", []),
            errors=state.get("errors", [])
        )
    
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    active_websockets[session_id] = websocket
    
    print(f"🔌 WebSocket connected: {session_id}")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            query = json.loads(data)["query"]
            
            print(f"💬 WS message: {query[:50]}...")
            
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
            
            print(f"✓ WS response sent")
    
    except WebSocketDisconnect:
        del active_websockets[session_id]
        print(f"🔌 WebSocket disconnected: {session_id}")


@app.get("/reports/{filename}")
async def download_report(filename: str):
    """Download generated report"""
    from config import REPORT_OUTPUT_DIR
    report_path = REPORT_OUTPUT_DIR / filename
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    print(f"📥 Downloading report: {filename}")
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
    print("\n" + "="*80)
    print("🚀 STARTING REAL ESTATE MULTI-AGENT SYSTEM")
    print("="*80)
    print("📍 API Server: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)