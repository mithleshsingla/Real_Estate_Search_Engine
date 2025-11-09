"""
Configuration for Multi-Agent System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent  # D:\mtech\Smartsense\agents
SMARTSENSE_DIR = BASE_DIR.parent  # D:\mtech\Smartsense
ASSETS_DIR = SMARTSENSE_DIR / "assets"
ETL_DIR = SMARTSENSE_DIR / "etl"
FLOORPLAN_PARSER_DIR = SMARTSENSE_DIR / "floorplan_parser"

# Data paths (from ETL)
EXCEL_FILE = ASSETS_DIR / "Property_list.xlsx"
IMAGES_DIR = ASSETS_DIR / "images"
CERTIFICATES_DIR = ASSETS_DIR / "certificates"

# Model paths (for floorplan parsing)
CHECKPOINT_PATH = FLOORPLAN_PARSER_DIR / "checkpoints" / "best.pth"
ROOM_CLASSIFIER_PATH = FLOORPLAN_PARSER_DIR / "models" / "room_classifier.pkl"

# API Keys
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "smartsense")

# Validate required API keys
REQUIRED_KEYS = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "TAVILY_API_KEY": TAVILY_API_KEY,
    "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY
}

for key_name, key_value in REQUIRED_KEYS.items():
    if not key_value:
        print(f"⚠️  WARNING: {key_name} not found in environment variables!")

# Database Configuration (reuse from ETL)
POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "user": "myuser",
    "password": "mypassword"
}

# Qdrant Configuration
QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "collection_name": "property_embeddings"
}

# LLM Configuration
GROQ_MODELS = {
    "router": "llama-3.3-70b-versatile",
    "sql": "llama-3.3-70b-versatile",
    "web": "llama-3.3-70b-versatile",  # Changed from mixtral (decommissioned)
    "renovation": "llama-3.3-70b-versatile"
}

GEMINI_MODELS = {
    "planner": "gemini-2.0-flash-exp",
    "rag": "gemini-2.0-flash-exp",
    "report": "gemini-1.5-pro"
}

# Embedding model (for RAG)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768
# Agent Configuration
ROUTER_TEMPERATURE = 0.1  # Low temperature for consistent routing
PLANNER_TEMPERATURE = 0.3  # Moderate for structured planning
RAG_TEMPERATURE = 0.5      # Higher for creative synthesis
REPORT_TEMPERATURE = 0.4   # Moderate for detailed reports

# Tavily Configuration
TAVILY_MAX_RESULTS = 5
TAVILY_SEARCH_DEPTH = "advanced"  # 'basic' or 'advanced'

# Memory Configuration
MAX_CONVERSATION_HISTORY = 10  # Keep last 10 exchanges
SESSION_TIMEOUT_MINUTES = 30

# Report Templates
REPORT_TEMPLATE_DIR = BASE_DIR / "templates"
REPORT_OUTPUT_DIR = BASE_DIR / "reports"

# Ensure directories exist
REPORT_TEMPLATE_DIR.mkdir(exist_ok=True)
REPORT_OUTPUT_DIR.mkdir(exist_ok=True)

# Renovation Cost Estimates (per sq ft in INR)
RENOVATION_COSTS = {
    "basic": {"min": 800, "max": 1200},
    "standard": {"min": 1200, "max": 1800},
    "premium": {"min": 1800, "max": 2500},
    "luxury": {"min": 2500, "max": 4000}
}

# Floorplan Parser Configuration
FLOORPLAN_CONFIDENCE_THRESHOLD = 0.3
ML_CONFIDENCE_THRESHOLD = 0.6

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Verify critical paths exist
def verify_paths():
    """Verify that critical paths exist"""
    critical_paths = {
        "Excel File": EXCEL_FILE,
        "Images Directory": IMAGES_DIR,
        "Certificates Directory": CERTIFICATES_DIR,
        "Checkpoint File": CHECKPOINT_PATH,
        "Room Classifier": ROOM_CLASSIFIER_PATH
    }
    
    missing_paths = []
    for name, path in critical_paths.items():
        if not path.exists():
            missing_paths.append(f"{name}: {path}")
    
    if missing_paths:
        print("\n⚠️  WARNING: Missing paths detected:")
        for missing in missing_paths:
            print(f"  - {missing}")
        print("\nSome features may not work correctly.\n")
    else:
        print("✓ All critical paths verified\n")

# Run verification on import (can be disabled)
if os.getenv("VERIFY_PATHS", "true").lower() == "true":
    verify_paths()