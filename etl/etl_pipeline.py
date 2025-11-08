"""
FastAPI ETL Pipeline for Property Data Ingestion
"""
import sys
from pathlib import Path

# Add floorplan_parser to Python path
sys.path.append(str(Path(__file__).parent.parent / "floorplan_parser"))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

from config import (
    EXCEL_FILE, IMAGES_DIR, CERTIFICATES_DIR,
    CHECKPOINT_PATH, ROOM_CLASSIFIER_PATH,
    FLOORPLAN_CONFIDENCE_THRESHOLD, ML_CONFIDENCE_THRESHOLD
)
from database import DatabaseHandler
from vector_store import VectorStore
from pdf_processor import PDFProcessor

# Import floorplan parser
from floorplan_inference import FloorplanParser


# FastAPI app
app = FastAPI(
    title="Property ETL Pipeline",
    description="ETL pipeline for property data with floorplan parsing and RAG",
    version="1.0.0"
)


# Pydantic models for API
class IngestionRequest(BaseModel):
    excel_path: Optional[str] = None
    recreate_collections: bool = False
    skip_existing: bool = False


class IngestionResponse(BaseModel):
    status: str
    message: str
    task_id: Optional[str] = None


class IngestionStatus(BaseModel):
    total_properties: int
    processed: int
    successful: int
    failed: int
    skipped: int
    failed_properties: List[Dict]
    processing_time: Optional[float] = None


# Global state for tracking ingestion status
ingestion_state = {
    "status": "idle",
    "current_task": None
}


class ETLPipeline:
    """Main ETL Pipeline orchestrator"""
    
    def __init__(self):
        self.db = DatabaseHandler()
        self.vector_store = VectorStore()
        self.pdf_processor = PDFProcessor()
        self.floorplan_parser = None
        
        # Statistics
        self.stats = {
            "total": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "failed_properties": []
        }
    
    def initialize_floorplan_parser(self):
        """Lazy initialization of floorplan parser (heavy model)"""
        if self.floorplan_parser is None:
            print("Initializing floorplan parser...")
            self.floorplan_parser = FloorplanParser(
                checkpoint_path=str(CHECKPOINT_PATH),
                confidence_threshold=FLOORPLAN_CONFIDENCE_THRESHOLD,
                use_ml_classifier=True,
                ml_classifier_path=str(ROOM_CLASSIFIER_PATH),
                ml_confidence_threshold=ML_CONFIDENCE_THRESHOLD
            )
            print("✓ Floorplan parser initialized")
    
    def parse_floorplan(self, image_path: Path) -> Optional[Dict]:
        """
        Parse floorplan image and return structured JSON
        
        Args:
            image_path: Path to floorplan image
            
        Returns:
            Parsed floorplan data or None if failed
        """
        try:
            self.initialize_floorplan_parser()
            output, detections = self.floorplan_parser.parse(str(image_path))
            return output
        except Exception as e:
            print(f"✗ Floorplan parsing failed for {image_path.name}: {e}")
            return None
    
    def process_certificates(self, cert_string: str) -> tuple[List[str], str]:
        """
        Process certificate column value
        
        Args:
            cert_string: Certificate string (e.g., "cert1.pdf|cert2.pdf")
            
        Returns:
            Tuple of (certificate_list, combined_text)
        """
        if pd.isna(cert_string) or not cert_string:
            return [], ""
        
        # Split by pipe separator
        cert_files = [c.strip() for c in cert_string.split('|')]
        
        # Extract text from PDFs
        cert_texts = self.pdf_processor.process_certificates(cert_files, CERTIFICATES_DIR)
        
        # Combine texts
        combined_text = self.pdf_processor.combine_certificate_texts(cert_texts)
        
        return cert_files, combined_text
    
    def process_single_property(self, row: pd.Series) -> Dict:
        """
        Process a single property row
        
        Args:
            row: Pandas Series representing one property
            
        Returns:
            Processed property data dictionary
        """
        property_id = row['property_id']
        print(f"\n{'='*60}")
        print(f"Processing: {property_id}")
        print(f"{'='*60}")
        
        # Parse floorplan
        image_path = IMAGES_DIR / row['image_file']
        floorplan_parsed = None
        
        if image_path.exists():
            print(f"📐 Parsing floorplan: {row['image_file']}")
            floorplan_parsed = self.parse_floorplan(image_path)
            if floorplan_parsed:
                print(f"✓ Floorplan parsed: {json.dumps(floorplan_parsed, indent=2)}")
            else:
                print(f"⚠ Floorplan parsing failed")
                self.stats['failed_properties'].append({
                    'property_id': property_id,
                    'reason': 'floorplan_parsing_failed'
                })
        else:
            print(f"⚠ Image not found: {image_path}")
            self.stats['failed_properties'].append({
                'property_id': property_id,
                'reason': 'image_not_found'
            })
        
        # Process certificates
        cert_files, cert_text = self.process_certificates(row.get('certificates'))
        if cert_files:
            print(f"📄 Processed {len(cert_files)} certificates")
        
        # Prepare property data
        property_data = {
            'property_id': property_id,
            'title': row['title'],
            'long_description': row['long_description'],
            'location': row['location'],
            'price': int(row['price']),
            'seller_type': row['seller_type'],
            'listing_date': row['listing_date'],
            'seller_contact': float(row['seller_contact']) if pd.notna(row['seller_contact']) else None,
            'metadata_tags': row['metadata_tags'],
            'image_file': row['image_file'],
            'floorplan_parsed': json.dumps(floorplan_parsed) if floorplan_parsed else None,
            'certificates': cert_files,
            'certificate_texts': cert_text
        }
        
        return property_data
    
    def run_ingestion(self, excel_path: Path = None, recreate_collections: bool = False) -> IngestionStatus:
        """
        Main ingestion pipeline
        
        Args:
            excel_path: Path to Excel file (default: from config)
            recreate_collections: Whether to recreate DB schema and vector collection
            
        Returns:
            IngestionStatus with results
        """
        start_time = datetime.now()
        excel_path = excel_path or EXCEL_FILE
        
        print(f"\n{'='*80}")
        print(f"STARTING ETL INGESTION PIPELINE")
        print(f"{'='*80}")
        print(f"Excel file: {excel_path}")
        print(f"Recreate collections: {recreate_collections}")
        
        # Step 1: Initialize database and vector store
        print(f"\n{'='*60}")
        print("STEP 1: Initialize Database and Vector Store")
        print(f"{'='*60}")
        
        try:
            self.db.create_schema()
            self.vector_store.create_collection(recreate=recreate_collections)
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Initialization failed: {e}")
        
        # Step 2: Load Excel data
        print(f"\n{'='*60}")
        print("STEP 2: Load Excel Data")
        print(f"{'='*60}")
        
        try:
            df = pd.read_excel(excel_path)
            self.stats['total'] = len(df)
            print(f"✓ Loaded {len(df)} properties from Excel")
        except Exception as e:
            print(f"✗ Failed to load Excel: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load Excel: {e}")
        
        # Step 3: Process each property
        print(f"\n{'='*60}")
        print("STEP 3: Process Properties")
        print(f"{'='*60}")
        
        for idx, row in df.iterrows():
            self.stats['processed'] += 1
            
            try:
                # Process property
                property_data = self.process_single_property(row)
                
                # Insert into PostgreSQL
                if self.db.insert_property(property_data):
                    print(f"✓ Saved to PostgreSQL")
                else:
                    print(f"✗ Failed to save to PostgreSQL")
                    self.stats['failed'] += 1
                    continue
                
                # Index in Qdrant
                if self.vector_store.index_property(property_data):
                    print(f"✓ Indexed in Qdrant")
                else:
                    print(f"⚠ Failed to index in Qdrant")
                
                self.stats['successful'] += 1
                print(f"✓ Property {property_data['property_id']} completed successfully")
                
            except Exception as e:
                print(f"✗ Error processing property: {e}")
                self.stats['failed'] += 1
                self.stats['failed_properties'].append({
                    'property_id': row['property_id'],
                    'reason': str(e)
                })
        
        # Final summary
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"INGESTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total properties: {self.stats['total']}")
        print(f"Processed: {self.stats['processed']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        if self.stats['failed_properties']:
            print(f"\n⚠ Failed Properties:")
            for failed in self.stats['failed_properties']:
                print(f"  - {failed['property_id']}: {failed['reason']}")
        
        return IngestionStatus(
            total_properties=self.stats['total'],
            processed=self.stats['processed'],
            successful=self.stats['successful'],
            failed=self.stats['failed'],
            skipped=self.stats['skipped'],
            failed_properties=self.stats['failed_properties'],
            processing_time=processing_time
        )


# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Property ETL Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ingest": "Trigger data ingestion",
            "GET /status": "Get ingestion status",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db = DatabaseHandler()
        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1")
        
        # Check Qdrant connection
        vector_store = VectorStore()
        vector_store.client.get_collections()
        
        return {
            "status": "healthy",
            "postgres": "connected",
            "qdrant": "connected"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/ingest", response_model=IngestionResponse)
async def trigger_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger ETL ingestion pipeline
    
    This endpoint starts the ingestion process in the background.
    Use /status endpoint to check progress.
    """
    if ingestion_state["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Ingestion already in progress"
        )
    
    # Update state
    ingestion_state["status"] = "running"
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ingestion_state["current_task"] = task_id
    
    # Run ingestion in background
    def run_pipeline():
        try:
            pipeline = ETLPipeline()
            excel_path = Path(request.excel_path) if request.excel_path else None
            result = pipeline.run_ingestion(
                excel_path=excel_path,
                recreate_collections=request.recreate_collections
            )
            ingestion_state["status"] = "completed"
            ingestion_state["result"] = result.dict()
        except Exception as e:
            ingestion_state["status"] = "failed"
            ingestion_state["error"] = str(e)
    
    background_tasks.add_task(run_pipeline)
    
    return IngestionResponse(
        status="started",
        message="Ingestion pipeline started in background",
        task_id=task_id
    )


@app.get("/status")
async def get_status():
    """Get current ingestion status"""
    return {
        "status": ingestion_state["status"],
        "task_id": ingestion_state.get("current_task"),
        "result": ingestion_state.get("result"),
        "error": ingestion_state.get("error")
    }


@app.get("/properties")
async def list_properties(limit: int = 10, offset: int = 0):
    """List all properties from database"""
    try:
        db = DatabaseHandler()
        properties = db.get_all_properties()
        return {
            "total": len(properties),
            "properties": properties[offset:offset+limit]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_properties(query: str, limit: int = 5):
    """Search properties using vector similarity"""
    try:
        vector_store = VectorStore()
        results = vector_store.search_similar(query, limit)
        return {
            "query": query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)