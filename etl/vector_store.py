"""
Qdrant vector database handler with embeddings
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
from config import QDRANT_CONFIG, EMBEDDING_MODEL, EMBEDDING_DIMENSION


class VectorStore:
    """Handle Qdrant vector database operations"""
    
    def __init__(self, config: dict = None, model_name: str = None):
        self.config = config or QDRANT_CONFIG
        self.model_name = model_name or EMBEDDING_MODEL
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.config["host"],
            port=self.config["port"]
        )
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.model_name}...")
        self.encoder = SentenceTransformer(self.model_name)
        print(f"✓ Embedding model loaded")
        
        self.collection_name = self.config["collection_name"]
    
    def create_collection(self, recreate: bool = False):
        """Create Qdrant collection if not exists"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists and recreate:
                self.client.delete_collection(self.collection_name)
                print(f"✓ Deleted existing collection: {self.collection_name}")
                exists = False
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                print(f"✓ Created collection: {self.collection_name}")
            else:
                print(f"✓ Collection already exists: {self.collection_name}")
                
        except Exception as e:
            print(f"✗ Error creating collection: {e}")
            raise
    
    def generate_embedding_text(self, property_data: Dict) -> str:
        """
        Generate text for embedding from property data
        
        Combines: title, description, floorplan data, metadata, certificates
        """
        parts = []
        
        # Title (weighted more)
        if property_data.get('title'):
            parts.append(f"Title: {property_data['title']}")
        
        # Long description
        if property_data.get('long_description'):
            parts.append(f"Description: {property_data['long_description']}")
        
        # Location
        if property_data.get('location'):
            parts.append(f"Location: {property_data['location']}")
        
        # Floorplan parsed data
        if property_data.get('floorplan_parsed'):
            fp = property_data['floorplan_parsed']
            if isinstance(fp, str):
                fp = json.loads(fp)
            
            # Add room details
            parts.append(f"Rooms: {fp.get('rooms', 0)}")
            parts.append(f"Bathrooms: {fp.get('bathrooms', 0)}")
            parts.append(f"Kitchens: {fp.get('kitchens', 0)}")
            
            # Add detailed room info
            if 'rooms_detail' in fp:
                room_info = []
                for room in fp['rooms_detail']:
                    label = room.get('label', '')
                    count = room.get('count', 0)
                    area = room.get('approx_area')
                    if area:
                        room_info.append(f"{label}: {count} ({area} sq units)")
                    else:
                        room_info.append(f"{label}: {count}")
                if room_info:
                    parts.append(f"Room Details: {', '.join(room_info)}")
        
        # Metadata tags
        if property_data.get('metadata_tags'):
            parts.append(f"Tags: {property_data['metadata_tags']}")
        
        # Certificate texts (inspection reports)
        if property_data.get('certificate_texts'):
            # Truncate if too long (keep first 2000 chars)
            cert_text = property_data['certificate_texts']
            if len(cert_text) > 2000:
                cert_text = cert_text[:2000] + "..."
            parts.append(f"Certificates: {cert_text}")
        
        return "\n".join(parts)
    
    def index_property(self, property_data: Dict) -> bool:
        """
        Index a single property in Qdrant
        
        Args:
            property_data: Property data dictionary
            
        Returns:
            Success status
        """
        try:
            # Generate embedding text
            embedding_text = self.generate_embedding_text(property_data)
            
            # Generate embedding vector
            embedding = self.encoder.encode(embedding_text).tolist()
            
            # Prepare payload (metadata to store with vector)
            payload = {
                "property_id": property_data["property_id"],
                "title": property_data.get("title", ""),
                "location": property_data.get("location", ""),
                "price": property_data.get("price", 0),
                "seller_type": property_data.get("seller_type", ""),
                "listing_date": str(property_data.get("listing_date", "")),
                "image_file": property_data.get("image_file", ""),
                "floorplan_parsed": property_data.get("floorplan_parsed", {}),
                "certificates": property_data.get("certificates", []),
                "metadata_tags": property_data.get("metadata_tags", "")
            }
            
            # Create point
            point = PointStruct(
                id=hash(property_data["property_id"]) & 0x7FFFFFFFFFFFFFFF,  # Positive int64
                vector=embedding,
                payload=payload
            )
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return True
            
        except Exception as e:
            print(f"✗ Error indexing property {property_data.get('property_id')}: {e}")
            return False
    
    def bulk_index_properties(self, properties: List[Dict]) -> int:
        """
        Bulk index multiple properties
        
        Args:
            properties: List of property data dictionaries
            
        Returns:
            Number of successfully indexed properties
        """
        success_count = 0
        
        for prop in properties:
            if self.index_property(prop):
                success_count += 1
        
        return success_count
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar properties using text query
        
        Args:
            query: Search query text
            limit: Number of results to return
            
        Returns:
            List of similar properties with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode(query).tolist()
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "property_id": result.payload["property_id"],
                    "title": result.payload["title"],
                    "score": result.score,
                    "location": result.payload["location"],
                    "price": result.payload["price"]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"✗ Error searching: {e}")
            return []
    
    def delete_collection(self):
        """Delete the collection (for testing)"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✓ Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"✗ Error deleting collection: {e}")