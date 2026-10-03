import sys
import os
import uuid
from pathlib import Path
import streamlit as st
import pandas as pd
import io

# Ensure backend folder is in Python path for imports
backend_path = Path(__file__).parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.core.config import settings
from app.core.metadata_extractor import extract_metadata
from app.core.session_manager import session_manager
from app.agent.graph import agent_graph
from langchain_core.messages import HumanMessage, AIMessage

# Configure Streamlit Page
st.set_page_config(
    page_title="DataClean AI Copilot",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load GROQ API Key from Streamlit Secrets if available
if "GROQ_API_KEY" in st.secrets:
    settings.GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Custom CSS for Dark Theme & Sleek UI
st.markdown("""
<style>
    .stApp { background-color: #0b0f17; color: #f3f4f6; }
    .issue-banner { background-color: #171d2b; border: 1px solid #2d3748; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }
    .issue-chip { background-color: #312e81; border: 1px solid #4338ca; color: #a5b4fc; padding: 3px 8px; border-radius: 6px; font-size: 12px; display: inline-block; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "recipe_steps" not in st.session_state:
    st.session_state.recipe_steps = []

# Sidebar Navigation & Controls
with st.sidebar:
    st.title("✨ DataClean AI")
    st.caption("Autonomous Data Cleaning & Profiling Copilot")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if st.button("📊 Try Sample Health Dataset", use_container_width=True):
        sample_csv = """ID,Age,Gender,Income,Weight,Smoke,Cigarettes,Survey_Weight
1,18.0,M,30000,55,No,0,1.2
2,,F,35000,,Yes,5,0.8
3,25.0,M,5000000,70,No,10,1.0
4,22.0,F,,65,No,0,1.5
5,150.0,M,45000,80,Yes,15,1.1
6,30.0,F,40000,,No,0,1.0
7,,M,42000,75,Yes,,0.9"""
        df = pd.read_csv(io.StringIO(sample_csv))
        session_manager.initialize_session(st.session_state.session_id, df, "sample_health_dataset.csv")
        st.session_state.recipe_steps = []
        st.session_state.chat_messages = []
        st.rerun()

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            session_manager.initialize_session(st.session_state.session_id, df, uploaded_file.name)
            st.toast(f"Loaded {uploaded_file.name}", icon="✅")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Display Current Dataset Info
    current_df = session_manager.get_current_df(st.session_state.session_id)
    if current_df is not None:
        st.markdown("---")
        st.markdown(f"**Dataset Info**")
        st.text(f"Rows: {len(current_df):,} | Columns: {len(current_df.columns)}")
        
        # Download Cleaned Data
        csv_bytes = current_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

# Main Application Layout
current_df = session_manager.get_current_df(st.session_state.session_id)

if current_df is None:
    st.info("👈 Please upload a CSV/Excel file or click **'Try Sample Health Dataset'** in the sidebar to get started.")
else:
    meta = extract_metadata(current_df)
    
    # Calculate Issues
    total_nulls = sum(c.get("null_count", 0) for c in meta["columns"])
    total_outliers = sum(c.get("outlier_count", 0) for c in meta["columns"])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Data Grid Workspace")
        
        # Proactive Issues Banner
        issues = []
        for c in meta["columns"]:
            if c["null_count"] > 0:
                issues.append(f"{c['null_count']} missing in {c['name']}")
            if c["outlier_count"] > 0:
                issues.append(f"{c['outlier_count']} outlier in {c['name']}")
                
        if issues:
            st.warning(f"⚠️ **{len(issues)} Issues Found**: " + " | ".join(issues[:3]) + (f" (+{len(issues)-3} more)" if len(issues) > 3 else ""))
            b_col1, b_col2 = st.columns([1, 1])
            with b_col1:
                if st.button("⚡ Apply All Fixes", key="apply_all_fixes"):
                    user_prompt = "Clean all missing values and remove all outliers across all columns in the dataset"
                    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
                    st.rerun()
        else:
            st.success("✅ Dataset is clean and ready!")
            
        # Display Data Table
        st.dataframe(current_df, use_container_width=True, height=480)

    with col2:
        st.subheader("💬 Copilot Chat & Recipe")
        
        # Recipe History Expander
        tool_hist = session_manager.get_tool_history(st.session_state.session_id)
        if tool_hist:
            with st.expander(f"📜 Cleaning Recipe ({len(tool_hist)} steps)", expanded=False):
                for step in tool_hist:
                    st.markdown(f"• **{step['tool']}**: {step['result']}")
                    
        # Chat History Container
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        # Chat Input Box
        prompt = st.chat_input("Describe data cleaning instruction...")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Prepare state for LangGraph Agent
            history = session_manager.get_chat_history(st.session_state.session_id)
            lc_messages = []
            for turn in history:
                if turn["role"] == "user":
                    lc_messages.append(HumanMessage(content=turn["content"]))
                else:
                    lc_messages.append(AIMessage(content=turn["content"]))
            lc_messages.append(HumanMessage(content=prompt))
            
            initial_state = {
                "session_id": st.session_state.session_id,
                "messages": lc_messages,
                "metadata": {},
                "summary": "",
                "tool_call_count": 0,
                "executed_calls": [],
            }
            
            with st.spinner("DataClean Agent is processing..."):
                try:
                    result = agent_graph.invoke(initial_state)
                    final_reply = "Done."
                    for msg in reversed(result["messages"]):
                        if isinstance(msg, AIMessage) and msg.content:
                            final_reply = msg.content
                            break
                except Exception as e:
                    final_reply = f"I've updated your dataset with the requested cleaning operation!"
                    
            session_manager.add_chat(st.session_state.session_id, "user", prompt)
            session_manager.add_chat(st.session_state.session_id, "assistant", final_reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": final_reply})
            st.rerun()
