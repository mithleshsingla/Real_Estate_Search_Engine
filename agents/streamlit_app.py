"""
Streamlit Frontend for Real Estate Multi-Agent System
"""
import streamlit as st
import requests
import json
import uuid
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Smart Real Estate Assistant",
    page_icon="🏠",
    layout="wide"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============= SIDEBAR =============
with st.sidebar:
    st.title("🏠 Smart Real Estate")
    st.write(f"**Session:** `{st.session_state.session_id[:8]}...`")
    
    page = st.selectbox(
        "Navigate",
        ["💬 Chat Assistant", "📤 Data Ingestion", "📐 Floorplan Parser"]
    )
    
    st.divider()
    
    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.caption("Powered by LangGraph Multi-Agent System")


# ============= PAGE 1: CHAT ASSISTANT =============
if page == "💬 Chat Assistant":
    st.title("💬 AI Real Estate Assistant")
    st.write("Ask me anything about properties, renovations, market rates, or generate reports!")
    
    # Example queries
    with st.expander("📝 Example Queries"):
        st.markdown("""
        - "Find 3BHK apartments in Mumbai under 50 lakh"
        - "Show me luxury properties near IT parks"
        - "Estimate renovation cost for a 2BHK apartment"
        - "What's the market rate in Bangalore?"
        - "Generate a detailed report for property PROP_123"
        - "Find properties with parking and balcony"
        """)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("ℹ️ Details"):
                        st.write(f"**Agents Used:** {', '.join(message['metadata'].get('agents', []))}")
                        
                        if message['metadata'].get('citations'):
                            st.write("**References:**")
                            for citation in message['metadata']['citations'][:5]:
                                st.caption(f"- {citation}")
    
    # Chat input
    user_query = st.chat_input("Ask about properties, renovations, market rates...")
    
    if user_query:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.write(user_query)
        
        # Show loading
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("🤖 AI agents are working..."):
                    try:
                        # Call API
                        response = requests.post(
                            f"{API_URL}/chat",
                            json={
                                "query": user_query,
                                "session_id": st.session_state.session_id
                            },
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Display response
                            st.write(data["response"])
                            
                            # Add to history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": data["response"],
                                "metadata": {
                                    "agents": data.get("executed_agents", []),
                                    "citations": data.get("citations", [])
                                }
                            })
                            
                            # Show metadata
                            with st.expander("ℹ️ Details"):
                                st.write(f"**Agents Used:** {', '.join(data.get('executed_agents', []))}")
                                
                                if data.get('citations'):
                                    st.write("**References:**")
                                    for citation in data['citations'][:5]:
                                        st.caption(f"- {citation}")
                            
                            st.rerun()
                        
                        else:
                            st.error(f"Error: {response.status_code}")
                    
                    except Exception as e:
                        st.error(f"Failed to get response: {str(e)}")


# ============= PAGE 2: DATA INGESTION =============
elif page == "📤 Data Ingestion":
    st.title("📤 Data Ingestion & ETL")
    st.write("Upload and process property data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Excel File Ingestion")
        
        st.info("""
        Upload an Excel file with property data. The system will:
        1. Parse floorplan images
        2. Extract certificate PDFs
        3. Store in PostgreSQL
        4. Index in Qdrant vector database
        """)
        
        excel_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
        
        recreate = st.checkbox("Recreate collections (fresh start)", value=False)
        
        if st.button("🚀 Start Ingestion", type="primary"):
            if excel_file:
                with st.spinner("Processing... This may take several minutes"):
                    try:
                        # Save file
                        temp_path = Path(f"/tmp/{excel_file.name}")
                        with open(temp_path, "wb") as f:
                            f.write(excel_file.read())
                        
                        # Trigger ingestion
                        response = requests.post(
                            f"{API_URL}/ingest",
                            json={
                                "excel_path": str(temp_path),
                                "recreate_collections": recreate
                            }
                        )
                        
                        if response.status_code == 200:
                            st.success("✅ Ingestion started! Check logs for progress.")
                        else:
                            st.error(f"Error: {response.json()}")
                    
                    except Exception as e:
                        st.error(f"Failed: {str(e)}")
            else:
                st.warning("Please upload an Excel file first")
    
    with col2:
        st.subheader("📈 Ingestion Status")
        
        if st.button("🔄 Refresh Status"):
            try:
                # Check health
                response = requests.get(f"{API_URL}/health")
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"**Status:** {data['status']}")
                    st.write(f"**PostgreSQL:** {data.get('postgres', 'unknown')}")
            except:
                st.error("Cannot connect to API")


# ============= PAGE 3: FLOORPLAN PARSER =============
elif page == "📐 Floorplan Parser":
    st.title("📐 Floorplan Image Parser")
    st.write("Upload a floorplan image to extract room information")
    
    st.info("""
    Upload a floorplan image and the AI will detect:
    - Number of bedrooms, bathrooms, kitchens
    - Room types and approximate areas
    - Layout details
    """)
    
    uploaded_file = st.file_uploader("Upload Floorplan Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(uploaded_file, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Parsed Results")
            
            if st.button("🚀 Parse Floorplan", type="primary"):
                with st.spinner("Parsing floorplan..."):
                    try:
                        # Call API
                        files = {"file": uploaded_file}
                        response = requests.post(f"{API_URL}/parse-floorplan", files=files)
                        
                        if response.status_code == 200:
                            data = response.json()
                            parsed = data["parsed_data"]
                            
                            # Display results
                            st.success("✅ Parsing complete!")
                            
                            # Metrics
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            with metrics_col1:
                                st.metric("🛏️ Bedrooms", parsed.get("rooms", 0))
                            with metrics_col2:
                                st.metric("🚿 Bathrooms", parsed.get("bathrooms", 0))
                            with metrics_col3:
                                st.metric("🍳 Kitchens", parsed.get("kitchens", 0))
                            
                            # Room details
                            if parsed.get("rooms_detail"):
                                st.write("**Room Details:**")
                                for room in parsed["rooms_detail"]:
                                    label = room.get("label", "Unknown")
                                    count = room.get("count", 0)
                                    area = room.get("approx_area")
                                    
                                    if area:
                                        st.write(f"- {label}: {count} ({area} sq units)")
                                    else:
                                        st.write(f"- {label}: {count}")
                            
                            # JSON output
                            with st.expander("📄 Full JSON Output"):
                                st.json(parsed)
                        
                        else:
                            st.error(f"Parsing failed: {response.json()}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


# ============= FOOTER =============
st.divider()
st.caption("🤖 Multi-Agent System powered by LangGraph | 🔗 Groq + Gemini + Tavily")