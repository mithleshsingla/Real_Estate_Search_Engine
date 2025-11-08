"""
PostgreSQL database tools for SQL Agent
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "etl"))

from database import DatabaseHandler
from typing import List, Dict, Optional
import json


class SQLTools:
    """Tools for querying PostgreSQL"""
    
    def __init__(self):
        self.db = DatabaseHandler()
    
    def search_properties(
        self,
        location: Optional[str] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        bedrooms: Optional[int] = None,
        property_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search properties with filters
        
        Args:
            location: City or area name
            min_price: Minimum price
            max_price: Maximum price
            bedrooms: Number of bedrooms
            property_type: Type of property
            limit: Max results
            
        Returns:
            List of matching properties
        """
        query_parts = ["SELECT * FROM properties WHERE 1=1"]
        params = []
        
        if location:
            query_parts.append("AND LOWER(location) LIKE LOWER(%s)")
            params.append(f"%{location}%")
        
        if min_price:
            query_parts.append("AND price >= %s")
            params.append(min_price)
        
        if max_price:
            query_parts.append("AND price <= %s")
            params.append(max_price)
        
        if bedrooms:
            query_parts.append("AND floorplan_parsed->>'rooms' = %s")
            params.append(str(bedrooms))
        
        if property_type:
            query_parts.append("AND LOWER(metadata_tags) LIKE LOWER(%s)")
            params.append(f"%{property_type}%")
        
        query_parts.append(f"ORDER BY listing_date DESC LIMIT {limit}")
        query = " ".join(query_parts)
        
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                results = []
                for row in rows:
                    prop = dict(zip(columns, row))
                    # Parse JSON fields
                    if prop.get('floorplan_parsed'):
                        if isinstance(prop['floorplan_parsed'], str):
                            prop['floorplan_parsed'] = json.loads(prop['floorplan_parsed'])
                    results.append(prop)
                
                return results
        
        except Exception as e:
            print(f"✗ SQL search error: {e}")
            return []
    
    def get_property_by_id(self, property_id: str) -> Optional[Dict]:
        """Get specific property by ID"""
        return self.db.get_property(property_id)
    
    def get_properties_by_ids(self, property_ids: List[str]) -> List[Dict]:
        """Get multiple properties by IDs"""
        results = []
        for pid in property_ids:
            prop = self.get_property_by_id(pid)
            if prop:
                results.append(prop)
        return results
    
    def aggregate_stats(
        self,
        location: Optional[str] = None
    ) -> Dict:
        """
        Get aggregate statistics
        
        Args:
            location: Filter by location
            
        Returns:
            Statistics dictionary
        """
        query = "SELECT COUNT(*) as total, AVG(price) as avg_price, MIN(price) as min_price, MAX(price) as max_price FROM properties"
        params = []
        
        if location:
            query += " WHERE LOWER(location) LIKE LOWER(%s)"
            params.append(f"%{location}%")
        
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                if result:
                    return {
                        "total_properties": result[0],
                        "avg_price": round(result[1], 2) if result[1] else 0,
                        "min_price": result[2] if result[2] else 0,
                        "max_price": result[3] if result[3] else 0,
                        "location": location
                    }
        
        except Exception as e:
            print(f"✗ Stats error: {e}")
        
        return {}
    
    def get_distinct_locations(self) -> List[str]:
        """Get all unique locations"""
        query = "SELECT DISTINCT location FROM properties ORDER BY location"
        
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(query)
                return [row[0] for row in cursor.fetchall()]
        except:
            return []