"""
Configuration for Multi-Agent System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
SMARTSENSE_DIR = BASE_DIR.parent
ASSETS_DIR = SMARTSENSE_DIR / "assets"
ETL_DIR = SMARTSENSE_DIR / "etl"

# API Keys
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "smartsense")

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
    "web": "mixtral-8x7b-32768",
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

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT