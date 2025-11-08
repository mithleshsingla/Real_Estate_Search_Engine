"""
Test script for ETL Pipeline
Run this to test the ingestion pipeline without FastAPI
"""
import sys
from pathlib import Path

# Add floorplan_parser to path
sys.path.append(str(Path(__file__).parent.parent / "floorplan_parser"))

from etl_pipeline import ETLPipeline
from config import EXCEL_FILE
import json


def test_connections():
    """Test database connections"""
    print("="*60)
    print("TESTING CONNECTIONS")
    print("="*60)
    
    # Test PostgreSQL
    try:
        from database import DatabaseHandler
        db = DatabaseHandler()
        with db.get_cursor() as cursor:
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            print(f"✓ PostgreSQL connected: {version[:50]}...")
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
        return False
    
    # Test Qdrant
    try:
        from vector_store import VectorStore
        vs = VectorStore()
        collections = vs.client.get_collections()
        print(f"✓ Qdrant connected: {len(collections.collections)} collections")
    except Exception as e:
        print(f"✗ Qdrant connection failed: {e}")
        return False
    
    return True


def test_single_property():
    """Test processing a single property"""
    print("\n" + "="*60)
    print("TESTING SINGLE PROPERTY")
    print("="*60)
    
    import pandas as pd
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        print(f"✓ Loaded Excel with {len(df)} properties")
        
        # Take first property
        row = df.iloc[0]
        print(f"\nTesting property: {row['property_id']}")
        
        pipeline = ETLPipeline()
        property_data = pipeline.process_single_property(row)
        
        print("\n✓ Property processed successfully!")
        print(json.dumps({
            'property_id': property_data['property_id'],
            'title': property_data['title'],
            'floorplan_parsed': property_data['floorplan_parsed'],
            'certificates': property_data['certificates']
        }, indent=2))
        
        return True
        
    except Exception as e:
        print(f"✗ Single property test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_ingestion():
    """Run full ingestion pipeline"""
    print("\n" + "="*60)
    print("RUNNING FULL INGESTION")
    print("="*60)
    
    response = input("This will process all properties. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    try:
        pipeline = ETLPipeline()
        result = pipeline.run_ingestion(recreate_collections=False)
        
        print("\n" + "="*60)
        print("INGESTION RESULTS")
        print("="*60)
        print(f"Total: {result.total_properties}")
        print(f"Successful: {result.successful}")
        print(f"Failed: {result.failed}")
        print(f"Time: {result.processing_time:.2f}s")
        
        if result.failed_properties:
            print(f"\nFailed properties:")
            for failed in result.failed_properties:
                print(f"  - {failed['property_id']}: {failed['reason']}")
        
    except Exception as e:
        print(f"✗ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()


def test_search():
    """Test vector search"""
    print("\n" + "="*60)
    print("TESTING VECTOR SEARCH")
    print("="*60)
    
    from vector_store import VectorStore
    
    try:
        vs = VectorStore()
        
        queries = [
            "3 bedroom apartment with parking",
            "luxury property near city center",
            "affordable house with garden"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            results = vs.search_similar(query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['title']} (score: {result['score']:.3f})")
                    print(f"     Location: {result['location']}, Price: {result['price']}")
            else:
                print("  No results found")
        
        print("\n✓ Search test completed")
        
    except Exception as e:
        print(f"✗ Search test failed: {e}")


def main():
    """Main test menu"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║          Property ETL Pipeline - Test Suite              ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\n" + "="*60)
        print("TEST MENU")
        print("="*60)
        print("1. Test connections (PostgreSQL + Qdrant)")
        print("2. Test single property processing")
        print("3. Run full ingestion pipeline")
        print("4. Test vector search")
        print("5. Exit")
        print("="*60)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            test_connections()
        elif choice == '2':
            if test_connections():
                test_single_property()
        elif choice == '3':
            if test_connections():
                run_full_ingestion()
        elif choice == '4':
            test_search()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main()