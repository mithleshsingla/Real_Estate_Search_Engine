"""
PostgreSQL database handler
"""
import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2.extensions import register_adapter
from contextlib import contextmanager
import json
from typing import List, Dict, Optional
from config import POSTGRES_CONFIG

# Register JSON adapter
register_adapter(dict, Json)


class DatabaseHandler:
    """Handle PostgreSQL operations"""
    
    def __init__(self, config: dict = None):
        self.config = config or POSTGRES_CONFIG
        self.connection = None
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        conn = psycopg2.connect(**self.config)
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def create_schema(self):
        """Create database schema if not exists"""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS properties (
            property_id VARCHAR(255) PRIMARY KEY,
            title TEXT NOT NULL,
            long_description TEXT,
            location TEXT,
            price INTEGER,
            seller_type VARCHAR(100),
            listing_date TIMESTAMP,
            seller_contact FLOAT,
            metadata_tags TEXT,
            image_file VARCHAR(500),
            floorplan_parsed JSONB,
            certificates TEXT[],
            certificate_texts TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_location ON properties(location);
        CREATE INDEX IF NOT EXISTS idx_price ON properties(price);
        CREATE INDEX IF NOT EXISTS idx_listing_date ON properties(listing_date);
        CREATE INDEX IF NOT EXISTS idx_floorplan_parsed ON properties USING GIN (floorplan_parsed);
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(schema_sql)
        print("✓ Database schema created/verified")
    
    def insert_property(self, property_data: Dict) -> bool:
        """Insert a single property record"""
        insert_sql = """
        INSERT INTO properties (
            property_id, title, long_description, location, price,
            seller_type, listing_date, seller_contact, metadata_tags,
            image_file, floorplan_parsed, certificates, certificate_texts
        ) VALUES (
            %(property_id)s, %(title)s, %(long_description)s, %(location)s, %(price)s,
            %(seller_type)s, %(listing_date)s, %(seller_contact)s, %(metadata_tags)s,
            %(image_file)s, %(floorplan_parsed)s, %(certificates)s, %(certificate_texts)s
        )
        ON CONFLICT (property_id) DO UPDATE SET
            title = EXCLUDED.title,
            long_description = EXCLUDED.long_description,
            location = EXCLUDED.location,
            price = EXCLUDED.price,
            seller_type = EXCLUDED.seller_type,
            listing_date = EXCLUDED.listing_date,
            seller_contact = EXCLUDED.seller_contact,
            metadata_tags = EXCLUDED.metadata_tags,
            image_file = EXCLUDED.image_file,
            floorplan_parsed = EXCLUDED.floorplan_parsed,
            certificates = EXCLUDED.certificates,
            certificate_texts = EXCLUDED.certificate_texts,
            updated_at = NOW()
        """
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute(insert_sql, property_data)
            return True
        except Exception as e:
            print(f"✗ Error inserting property {property_data.get('property_id')}: {e}")
            return False
    
    def bulk_insert_properties(self, properties: List[Dict]) -> int:
        """Bulk insert multiple properties"""
        success_count = 0
        for prop in properties:
            if self.insert_property(prop):
                success_count += 1
        return success_count
    
    def get_property(self, property_id: str) -> Optional[Dict]:
        """Retrieve a property by ID"""
        query_sql = "SELECT * FROM properties WHERE property_id = %s"
        
        with self.get_cursor() as cursor:
            cursor.execute(query_sql, (property_id,))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        return None
    
    def get_all_properties(self) -> List[Dict]:
        """Retrieve all properties"""
        query_sql = "SELECT * FROM properties ORDER BY listing_date DESC"
        
        with self.get_cursor() as cursor:
            cursor.execute(query_sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    def delete_all_properties(self):
        """Delete all properties (for testing)"""
        with self.get_cursor() as cursor:
            cursor.execute("DELETE FROM properties")
        print("✓ All properties deleted")