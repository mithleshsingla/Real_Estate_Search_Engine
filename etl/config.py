"""
Configuration file for ETL Pipeline
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(r"D:\mtech\Smartsense")
ASSETS_DIR = BASE_DIR / "assets"
FLOORPLAN_PARSER_DIR = BASE_DIR / "floorplan_parser"

# Data paths
EXCEL_FILE = ASSETS_DIR / "Property_list.xlsx"
IMAGES_DIR = ASSETS_DIR / "images"
CERTIFICATES_DIR = ASSETS_DIR / "certificates"

# Model paths
CHECKPOINT_PATH = FLOORPLAN_PARSER_DIR / "checkpoints" / "best.pth"
ROOM_CLASSIFIER_PATH = FLOORPLAN_PARSER_DIR / "models" / "room_classifier.pkl"

# Database configuration
POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "user": "myuser",
    "password": "mypassword"
}

# Qdrant configuration
QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "collection_name": "property_embeddings"
}

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768  # all-mpnet-base-v2 produces 768-dim vectors

# Floorplan parser settings
FLOORPLAN_CONFIDENCE_THRESHOLD = 0.3
ML_CONFIDENCE_THRESHOLD = 0.6

# Processing settings
BATCH_SIZE = 10  # Process images in batches to avoid memory issues