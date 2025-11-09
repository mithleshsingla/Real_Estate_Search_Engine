# 🏡 Smart Real Estate Multi-Agent System

> **AI-powered property search platform with floorplan parsing, ETL pipeline, RAG-enabled chatbot, and multi-agent orchestration using LangGraph**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Components](#-components)
  - [Floorplan Parser](#1-floorplan-parser)
  - [ETL Pipeline](#2-etl-pipeline)
  - [Multi-Agent System](#3-multi-agent-system)
  - [FastAPI Backend](#4-fastapi-backend)
  - [Streamlit Frontend](#5-streamlit-frontend)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **Smart Real Estate Multi-Agent System** is an end-to-end AI-powered platform that revolutionizes property search and analysis. It combines:

1. **Computer Vision** - Automated floorplan parsing using custom CNN models
2. **ETL Pipeline** - Structured and unstructured data ingestion with vector embeddings
3. **Multi-Agent AI** - LangGraph-orchestrated agents for complex query handling
4. **RAG System** - Retrieval-Augmented Generation for contextual property search
5. **Web Interface** - User-friendly Streamlit chat interface

**What makes it unique:**
- Automatically parses floorplan images to extract room information
- Hybrid search (SQL + Vector DB) for precise and semantic property matching
- Real-time market research via web search integration
- Renovation cost estimation with AI
- PDF report generation with market insights
- Persistent conversation memory across sessions

---

## ✨ Key Features

### 🔍 **Intelligent Property Search**
- **Structured Queries** - SQL-based filtering (location, price, bedrooms)
- **Semantic Search** - Vector similarity for fuzzy queries ("near tech parks")
- **Hybrid Approach** - Combines SQL + RAG for comprehensive results
- **Academic Citations** - Proper source attribution in responses

### 🏗️ **Floorplan Parsing**
- **Computer Vision Model** - Custom ResNet18 + FPN architecture
- **OCR Integration** - EasyOCR for text extraction
- **Room Classification** - Hybrid keyword + ML approach
- **Structured Output** - JSON format with room counts and areas

### 📊 **ETL Pipeline**
- **Excel Ingestion** - Automated data loading from Excel files
- **Image Processing** - Floorplan parsing for each property
- **PDF Extraction** - Certificate text extraction using PyMuPDF
- **Vector Indexing** - Qdrant vector database for semantic search
- **PostgreSQL Storage** - Structured data with JSONB support

### 🤖 **8 Specialized AI Agents**

| Agent | LLM | Purpose |
|-------|-----|---------|
| **Query Router** | Groq (Llama 3.3) | Intent detection & slot extraction |
| **Task Planner** | Gemini 2.0 | Multi-step query decomposition |
| **SQL Agent** | Groq (Llama 3.3) | Structured database queries |
| **RAG Agent** | Gemini 2.0 | Vector search & synthesis |
| **Web Research** | Groq + Tavily | Live market data & trends |
| **Renovation Estimator** | Groq (Llama 3.3) | Cost calculation with LLM |
| **Report Generator** | Gemini 1.5 Pro | HTML/PDF reports |
| **Memory Manager** | PostgreSQL | Session persistence |

### 💬 **Chat Interface**
- **Natural Language** - Conversational property search
- **Multi-turn Context** - Remembers user preferences
- **Real-time Updates** - WebSocket support for streaming
- **Citation Display** - Shows data sources with academic format

### 📈 **Market Intelligence**
- **Live Data** - Tavily web search for current market rates
- **Neighborhood Info** - Schools, hospitals, amenities
- **Price Trends** - Historical and current pricing data
- **Comparative Analysis** - Property comparisons

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         Streamlit Frontend (streamlit_app.py)             │  │
│  │  • Chat Interface  • Data Ingestion  • Floorplan Debug   │  │
│  └───────────────────────┬───────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────┘
                             │ HTTP/WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│                    FastAPI BACKEND (api.py)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ /chat       │  │ /ingest      │  │ /parse-floorplan   │    │
│  │ (REST/WS)   │  │ (ETL)        │  │ (Debug)            │    │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬──────────┘    │
└─────────┼────────────────┼─────────────────────┼───────────────┘
          │                │                     │
┌─────────▼────────────────▼─────────────────────▼───────────────┐
│               MULTI-AGENT ORCHESTRATION (graph.py)              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    LangGraph Workflow                     │  │
│  │                                                           │  │
│  │  ┌────────┐    ┌─────────┐    ┌──────────────────────┐ │  │
│  │  │ Router │───▶│ Planner │───▶│  Parallel Execution  │ │  │
│  │  └────────┘    └─────────┘    │  ┌────┐ ┌────┐ ┌───┐ │ │  │
│  │                                │  │SQL │ │RAG │ │Web│ │ │  │
│  │                                │  └────┘ └────┘ └───┘ │ │  │
│  │                                └──────────┬───────────┘ │  │
│  │                                           ▼             │  │
│  │               ┌───────────────────────────────┐        │  │
│  │               │   Aggregator + Memory         │        │  │
│  │               └───────────────────────────────┘        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│  PostgreSQL  │    │  Qdrant Vector │    │   Tavily     │
│  (Structured)│    │  DB (Semantic) │    │  (Web Search)│
│              │    │                │    │              │
│ • Properties │    │ • Embeddings   │    │ • Market     │
│ • Memory     │    │ • Citations    │    │ • Trends     │
│ • Sessions   │    │ • RAG Context  │    │ • News       │
└──────────────┘    └────────────────┘    └──────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               FLOORPLAN PROCESSING PIPELINE                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Image → ResNet18+FPN → OCR → Room Classifier → JSON    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### **AI & ML**
- **LangGraph** - Multi-agent orchestration
- **LangChain** - LLM framework
- **Groq** - Fast inference (Llama 3.3)
- **Google Gemini** - Advanced reasoning (Gemini 2.0)
- **PyTorch** - Deep learning framework
- **Sentence Transformers** - Text embeddings

### **Databases**
- **PostgreSQL** - Structured data storage
- **Qdrant** - Vector database for RAG
- **SQLite** - LangGraph checkpointing

### **Web & API**
- **FastAPI** - Backend API framework
- **Streamlit** - Interactive frontend
- **Uvicorn** - ASGI server
- **WebSockets** - Real-time communication

### **Computer Vision**
- **OpenCV** - Image processing
- **EasyOCR** - Optical character recognition
- **Albumentations** - Data augmentation
- **Pillow** - Image manipulation

### **Search & Research**
- **Tavily** - Web search API
- **EasyOCR** - Text extraction
- **PyMuPDF** - PDF text extraction

### **DevOps**
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **TensorBoard** - Training visualization

---

## 📁 Project Structure

```
D:\mtech\Smartsense\
│
├── 📐 floorplan_parser/              # Computer Vision Module
│   ├── floorplan_dataloader.py      # COCO data loading
│   ├── floorplan_model.py            # ResNet18 + FPN model
│   ├── floorplan_trainer.py          # Training script
│   ├── floorplan_inference.py        # Inference pipeline
│   ├── room_classifier.py            # ML room classifier
│   ├── checkpoints/
│   │   └── best.pth                 # Trained model weights
│   └── models/
│       └── room_classifier.pkl      # ML classifier
│
├── 📊 etl/                           # ETL Pipeline
│   ├── config.py                    # Configuration
│   ├── database.py                  # PostgreSQL handler
│   ├── vector_store.py              # Qdrant handler
│   ├── pdf_processor.py             # PDF text extraction
│   ├── etl_pipeline.py              # Main ETL logic
│   ├── test_pipeline.py             # Testing utilities
│   └── requirements.txt             # ETL dependencies
│
├── 🤖 agents/                        # Multi-Agent System
│   ├── agents/
│   │   ├── config.py                # Agent configuration
│   │   ├── state.py                 # Shared state schema
│   │   ├── router.py                # Query Router agent
│   │   ├── planner.py               # Task Planner agent
│   │   ├── sql_agent.py             # SQL Agent
│   │   ├── rag_agent.py             # RAG Agent
│   │   ├── web_agent.py             # Web Research agent
│   │   ├── renovation_agent.py      # Renovation Estimator
│   │   ├── report_agent.py          # Report + Memory
│   │   └── graph.py                 # LangGraph workflow
│   ├── tools/
│   │   └── sql_tools.py             # Database tools
│   ├── templates/                   # Report templates
│   ├── reports/                     # Generated reports
│   ├── api.py                       # FastAPI backend
│   ├── streamlit_app.py             # Frontend UI
│   ├── requirements.txt             # Agent dependencies
│   └── .env                         # API keys (not in repo)
│
├── 🗄️ rag_storage/                  # Database Services
│   └── docker-compose.yaml          # PostgreSQL + Qdrant
│
├── 📁 assets/                        # Data Assets
│   ├── Property_list.xlsx           # Property dataset
│   ├── images/                      # Floorplan images
│   └── certificates/                # PDF certificates
│
├── 📚 docs/                          # Documentation
│   ├── SETUP_GUIDE.md               # Setup instructions
│   ├── QUICKSTART.md                # Quick start guide
│   ├── INFINITE_LOOP_FIX.md         # Bug fixes
│   └── CRITICAL_FIXES_V2.md         # Latest fixes
│
└── README.md                         # This file
```

---

## 🔧 Installation

### Prerequisites

- **Python 3.8+** 
- **Docker & Docker Compose** (for PostgreSQL + Qdrant)
- **CUDA-capable GPU** (recommended for floorplan parsing)
- **16GB+ RAM**
- **20GB+ disk space**

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/smart-real-estate.git
cd smart-real-estate
```

### Step 2: Set Up Environment

```bash
# Create virtual environment
python -m venv smartsense-env

# Activate (Windows)
smartsense-env\Scripts\activate

# Activate (Linux/Mac)
source smartsense-env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r agents/requirements.txt
pip install -r etl/requirements.txt

# Or install PyTorch separately (adjust CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Set Up Databases

```bash
cd rag_storage
docker-compose up -d

# Verify services
docker-compose ps
# Should show postgres_db and qdrant_db running
```

### Step 5: Configure API Keys

Create `agents/.env` file:

```env
LANGCHAIN_API_KEY=your_langchain_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_PROJECT=smartsense
```

Get your API keys:
- **LangChain**: https://smith.langchain.com/
- **Groq**: https://console.groq.com/
- **Google AI**: https://makersuite.google.com/app/apikey
- **Tavily**: https://tavily.com/

### Step 6: Verify Installation

```bash
# Test database connections
cd agents
python -c "from agents.config import verify_paths; verify_paths()"

# Test agent system
cd agents/agents
python graph.py
```

---

## 🚀 Quick Start

### **Option A: Full Pipeline (Recommended)**

```bash
# 1. Start databases
cd rag_storage
docker-compose up -d

# 2. Ingest property data (ETL)
cd ../etl
python test_pipeline.py
# Select option 3: Run full ingestion

# 3. Start FastAPI backend
cd ../agents
python api.py

# 4. Start Streamlit frontend (new terminal)
streamlit run streamlit_app.py
```

Then open:
- **API Docs**: http://localhost:8000/docs
- **Chat UI**: http://localhost:8501

### **Option B: Test Individual Components**

#### Test Floorplan Parser

```bash
cd floorplan_parser
python floorplan_inference.py
```

#### Test ETL Pipeline

```bash
cd etl
python test_pipeline.py
# Select option 2: Test single property
```

#### Test Agent System

```bash
cd agents/agents
python graph.py
```

---

## 🧩 Components

### 1. **Floorplan Parser**

**Purpose**: Automatically extract room information from floorplan images.

**Key Files**:
- `floorplan_model.py` - ResNet18 + FPN architecture
- `floorplan_inference.py` - Inference pipeline with OCR
- `room_classifier.py` - Hybrid room classification

**Usage**:

```python
from floorplan_inference import FloorplanParser

parser = FloorplanParser(
    checkpoint_path='./checkpoints/best.pth',
    confidence_threshold=0.3
)

output, detections = parser.parse(
    image_path='floorplan.jpg',
    save_visualization='output.jpg'
)

print(output)
# {
#   "rooms": 3,
#   "bathrooms": 2,
#   "kitchens": 1,
#   "halls": 1,
#   "rooms_detail": [...]
# }
```

**Training**:

```bash
python floorplan_trainer.py
# Trains for 100 epochs, saves checkpoints
# Monitor with: tensorboard --logdir=./logs
```

---

### 2. **ETL Pipeline**

**Purpose**: Ingest property data from Excel, parse floorplans, extract PDFs, and index for search.

**Key Files**:
- `etl_pipeline.py` - Main ingestion logic
- `database.py` - PostgreSQL operations
- `vector_store.py` - Qdrant vector indexing
- `pdf_processor.py` - Certificate text extraction

**Usage**:

```python
from etl_pipeline import ETLPipeline

pipeline = ETLPipeline()
result = pipeline.run_ingestion(
    excel_path='assets/Property_list.xlsx',
    recreate_collections=False
)

print(f"Processed: {result.successful}/{result.total_properties}")
```

**What Gets Indexed**:
- Property title, description, location
- Floorplan parsed data (rooms, bathrooms, etc.)
- Certificate text content
- Metadata tags
- Vector embeddings for semantic search

---

### 3. **Multi-Agent System**

**Purpose**: Orchestrate 8 specialized AI agents to handle complex property queries.

**Key Files**:
- `graph.py` - LangGraph workflow
- `router.py` - Intent classification
- `planner.py` - Task decomposition
- `sql_agent.py` - Database queries
- `rag_agent.py` - Vector search
- `web_agent.py` - Market research
- `renovation_agent.py` - Cost estimation
- `report_agent.py` - PDF generation + memory

**Agent Flow**:

```
User Query
    ↓
Router (classify intent)
    ↓
Planner (decompose into tasks)
    ↓
┌─────────┬─────────┬─────────┐
│   SQL   │   RAG   │   Web   │ (Parallel)
└────┬────┴────┬────┴────┬────┘
     └─────────┴─────────┘
              ↓
         Aggregator
              ↓
           Memory
              ↓
         Response
```

**Example Query Handling**:

```python
from graph import run_agent_system

response, state = run_agent_system(
    query="Find 3BHK apartments in Mumbai under 50 lakh, check market rates, estimate renovation",
    session_id="user_123"
)

print(response)
# Found 5 properties matching your criteria.
# 
# **Market Insights:**
# Current market rates in Mumbai: ₹12,000-15,000 per sq ft...
#
# **Renovation Estimate:** ₹8,50,000
#
# **References:**
# - Luxury Apartment in Bandra (PROP_001, 2024)
# - Modern Flat in Andheri (PROP_002, 2024)
```

---

### 4. **FastAPI Backend**

**Purpose**: REST API and WebSocket endpoints for all functionality.

**Key Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/ingest` | POST | Trigger ETL pipeline |
| `/parse-floorplan` | POST | Parse single image |
| `/chat` | POST | Chat with AI agents (REST) |
| `/ws/chat/{session_id}` | WS | Real-time chat (WebSocket) |
| `/reports/{filename}` | GET | Download generated reports |
| `/properties` | GET | List all properties |
| `/search` | GET | Vector similarity search |

**Usage**:

```python
import requests

# Chat with agents
response = requests.post('http://localhost:8000/chat', json={
    'query': 'Show me 2BHK apartments in Bangalore',
    'session_id': 'user_123'
})

data = response.json()
print(data['response'])
print(data['executed_agents'])  # ['router', 'planner', 'sql_agent', ...]
print(data['citations'])  # Academic citations
```

---

### 5. **Streamlit Frontend**

**Purpose**: User-friendly web interface for all features.

**Pages**:

1. **💬 Chat Assistant**
   - Natural language property search
   - Shows agent execution details
   - Displays citations
   - Conversation history

2. **📤 Data Ingestion**
   - Upload Excel files
   - Trigger ETL pipeline
   - Monitor ingestion status

3. **📐 Floorplan Parser**
   - Upload single floorplan image
   - View parsed results
   - JSON output display

**Features**:
- Session management
- Real-time streaming (WebSocket)
- Example query suggestions
- Metadata display (agents used, citations)
- Error handling

---

## 📖 Usage Guide

### **Scenario 1: Simple Property Search**

```bash
# Via Streamlit
Query: "Find 3BHK apartments in Mumbai under 50 lakh"

# Via API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Find 3BHK apartments in Mumbai under 50 lakh"}'
```

**Agent Flow**: Router → Planner → SQL Agent → Response  
**Time**: ~5 seconds

---

### **Scenario 2: Complex Multi-Agent Query**

```bash
Query: "Show luxury properties near IT parks with good schools, check current market rates, and estimate renovation cost"
```

**Agent Flow**:
1. Router → SEMANTIC_SEARCH
2. Planner → [SQL_SEARCH, RAG_SEARCH, WEB_RESEARCH, RENOVATION_ESTIMATE]
3. SQL + RAG + Web agents run in parallel
4. Renovation agent processes results
5. Aggregator combines all data
6. Memory saves conversation

**Time**: ~15-20 seconds

---

### **Scenario 3: Report Generation**

```bash
Query: "Generate a detailed report for property PROP_123"
```

**Agent Flow**: Router → Planner → SQL + RAG + Web → Report Generator  
**Output**: HTML report at `/reports/report_20241109_143022.html`  
**Time**: ~20-25 seconds

---

### **Scenario 4: Data Ingestion**

```bash
# Via Streamlit: Upload Excel file in Data Ingestion page

# Via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"excel_path": "/path/to/Property_list.xlsx", "recreate_collections": false}'
```

**Process**:
1. Load Excel file (73 properties)
2. Parse each floorplan image (~5-10 sec/image)
3. Extract PDF certificates (~1-3 sec/cert)
4. Store in PostgreSQL
5. Generate embeddings
6. Index in Qdrant

**Total Time**: ~15-25 minutes for 73 properties

---

### **Scenario 5: Floorplan Parsing**

```bash
# Via Streamlit: Upload image in Floorplan Parser page

# Via API
curl -X POST http://localhost:8000/parse-floorplan \
  -F "file=@floorplan.jpg"
```

**Output**:
```json
{
  "filename": "floorplan.jpg",
  "parsed_data": {
    "rooms": 3,
    "bathrooms": 2,
    "kitchens": 1,
    "halls": 1,
    "rooms_detail": [...]
  },
  "detections_count": 15
}
```

---

## 📡 API Documentation

### **Authentication**

Currently no authentication required. Add JWT/OAuth for production.

### **REST Endpoints**

#### **POST /chat**

Chat with multi-agent system (synchronous).

**Request**:
```json
{
  "query": "Find 2BHK in Bangalore",
  "session_id": "optional-session-id"
}
```

**Response**:
```json
{
  "response": "Found 8 properties matching...",
  "session_id": "abc-123-def",
  "executed_agents": ["router", "planner", "sql_agent", "aggregator"],
  "citations": ["Property X in Y (PROP_001, 2024)"],
  "errors": []
}
```

#### **GET /search**

Vector similarity search.

**Request**:
```
GET /search?query=luxury%20apartment&limit=5
```

**Response**:
```json
{
  "query": "luxury apartment",
  "results": [
    {
      "property_id": "PROP_001",
      "title": "Luxury Apartment",
      "score": 0.92,
      "location": "Mumbai",
      "price": 5000000
    }
  ]
}
```

#### **POST /ingest**

Trigger ETL pipeline (background task).

**Request**:
```json
{
  "excel_path": "/path/to/file.xlsx",
  "recreate_collections": false
}
```

**Response**:
```json
{
  "status": "started",
  "message": "ETL pipeline started in background"
}
```

---

### **WebSocket Endpoint**

#### **WS /ws/chat/{session_id}**

Real-time chat with streaming responses.

**Connect**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat/user_123');
```

**Send**:
```json
{"query": "Show me 3BHK apartments"}
```

**Receive**:
```json
{"type": "typing", "message": "Agent system processing..."}
{"type": "response", "message": "Found 5 properties...", "executed_agents": [...]}
```

---

## ⚙️ Configuration

### **Agent Configuration** (`agents/config.py`)

```python
# LLM Models
GROQ_MODELS = {
    "router": "llama-3.3-70b-versatile",
    "sql": "llama-3.3-70b-versatile",
    "web": "llama-3.3-70b-versatile",
    "renovation": "llama-3.3-70b-versatile"
}

GEMINI_MODELS = {
    "planner": "gemini-2.0-flash-exp",
    "rag": "gemini-2.0-flash-exp",
    "report": "gemini-1.5-pro"
}

# Agent Retry Configuration
MAX_AGENT_RETRIES = 3  # Max retries per agent
ROUTER_TEMPERATURE = 0.1
PLANNER_TEMPERATURE = 0.3
RAG_TEMPERATURE = 0.5
```

### **Database Configuration** (`etl/config.py`)

```python
POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "user": "myuser",
    "password": "mypassword"
}

QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "collection_name": "property_embeddings"
}
```

### **Floorplan Parser Configuration**

```python
# In floorplan_inference.py
parser = FloorplanParser(
    checkpoint_path='./checkpoints/best.pth',
    confidence_threshold=0.3,  # 0.2-0.7 range
    use_ml_classifier=True,
    ml_confidence_threshold=0.6
)
```

---

## 🚢 Deployment

### **Production Deployment**

#### **1. Docker Deployment**

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -r agents/requirements.txt
RUN pip install -r etl/requirements.txt

EXPOSE 8000 8501

CMD ["uvicorn", "agents.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t smart-real-estate .
docker run -p 8000:8000 -p 8501:8501 smart-real-estate
```

#### **2. Environment Variables**

```bash
# Production .env
LANGCHAIN_API_KEY=prod_key
GROQ_API_KEY=prod_key
GOOGLE_API_KEY=prod_key
TAVILY_API_KEY=prod_key

# Database URLs (cloud)
POSTGRES_URL=postgresql://user:pass@host:5432/db
QDRANT_URL=https://qdrant-cloud-instance
```

#### **3. Scaling**

- **Load Balancer** - NGINX for API
- **Caching** - Redis for frequent queries
- **Queue** - Celery for background ETL tasks
- **CDN** - CloudFront for static assets

---

## 🔍 Troubleshooting

### **Common Issues**

#### 1. "GraphRecursionError: Recursion limit reached"

**Cause**: Agent retry logic issue  
**Fix**: Already fixed in latest `graph.py`. Ensure you have v2 fixes.

#### 2. "Model decommissioned: mixtral-8x7b-32768"

**Cause**: Groq deprecated Mixtral  
**Fix**: Update `config.py`:
```python
"web": "llama-3.3-70b-versatile"  # Not mixtral
```

#### 3. "Foreign key constraint: session_id not found"

**Cause**: Memory trying to insert before session creation