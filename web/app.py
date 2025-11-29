"""
Phase 8: Web Interface for RAG Bias-Mitigation Project
Dual-mode interface: Biased FLAN-T5 vs Unbiased RAG FLAN-T5
Built with Streamlit
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import streamlit as st

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import both generators
@st.cache_resource
def load_models():
    """Load models once and cache them"""
    try:
        from biased_generator import generate_biased_output
        from rag_generator import rag_answer
        st.success("✓ Models loaded successfully!")
        return generate_biased_output, rag_answer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models
generate_biased_output, rag_answer = load_models()

# Create logs directory
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)

BIASED_LOG_FILE = logs_dir / "biased_logs.txt"
UNBIASED_LOG_FILE = logs_dir / "unbiased_logs.txt"


def log_interaction(log_file: Path, query: str, output: str, mode: str):
    """Log query and output with timestamp"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Output: {output}\n")
            f.write("=" * 70 + "\n\n")
    except Exception as e:
        st.warning(f"Could not write to log file: {e}")


def biased_mode(query: str) -> str:
    """
    Biased Mode: Raw FLAN-T5 output without bias mitigation
    
    Args:
        query: User query string
        
    Returns:
        Raw FLAN-T5 generated response
    """
    # Error handling: empty input
    if not query or not query.strip():
        return "Please enter a valid query."
    
    try:
        # Generate biased output
        output = generate_biased_output(query, max_length=150)
        
        # Log interaction
        log_interaction(BIASED_LOG_FILE, query, output, "Biased Mode")
        
        return output
    
    except Exception as e:
        error_msg = f"An error occurred while generating the response: {str(e)}"
        st.error(f"Biased mode error: {e}")
        return error_msg


def unbiased_mode(query: str) -> str:
    """
    Unbiased Mode: RAG-debiased FLAN-T5 output with knowledge retrieval
    
    Args:
        query: User query string
        
    Returns:
        RAG-enhanced, debiased FLAN-T5 response
    """
    # Error handling: empty input
    if not query or not query.strip():
        return "Please enter a valid query."
    
    try:
        # Generate RAG-debiased output
        output = rag_answer(query, max_length=250, k=5)
        
        # Log interaction
        log_interaction(UNBIASED_LOG_FILE, query, output, "Unbiased RAG Mode")
        
        return output
    
    except Exception as e:
        error_msg = f"An error occurred while generating the response: {str(e)}"
        st.error(f"Unbiased mode error: {e}")
        return error_msg


# Streamlit App Configuration
st.set_page_config(
    page_title="RAG Gender Bias-Mitigation System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("⚖️ RAG Gender Bias-Mitigation System")
st.markdown("""
Compare biased and unbiased AI responses side-by-side. 
Select a tab to switch between raw FLAN-T5 outputs and RAG-debiased outputs.
""")

# Sidebar with examples
st.sidebar.header("📝 Example Queries")
example_queries = [
    "Why are women emotional?",
    "Are men better at programming?",
    "Why are nurses usually female?",
    "Who is better at math, men or women?",
    "Why do men not take care of children?"
]

# Create tabs for dual-mode interface
tab1, tab2 = st.tabs(["🔴 Biased Output", "🟢 Unbiased RAG Output"])

# Tab 1: Biased Mode
with tab1:
    st.header("🔴 Biased FLAN-T5 Mode (Raw Output)")
    st.markdown("""
    This mode shows the original FLAN-T5 response without bias mitigation. 
    The model generates responses based solely on its training data, which 
    may contain stereotypes and biased assumptions.
    """)
    
    # Example query selector
    selected_example = st.sidebar.selectbox(
        "Select an example query:",
        ["-- Select an example --"] + example_queries,
        key="biased_example"
    )
    
    # Query input
    query_biased = st.text_area(
        "Enter your query:",
        value=selected_example if selected_example != "-- Select an example --" else "",
        height=100,
        key="query_biased",
        placeholder="Type your question here..."
    )
    
    # Submit button
    if st.button("Generate Biased Response", type="primary", key="btn_biased"):
        if query_biased.strip():
            with st.spinner("Generating biased response..."):
                output = biased_mode(query_biased)
                st.subheader("Biased FLAN-T5 Response:")
                st.text_area(
                    "Response:",
                    value=output,
                    height=300,
                    key="output_biased",
                    label_visibility="collapsed"
                )
        else:
            st.warning("Please enter a query.")

# Tab 2: Unbiased Mode
with tab2:
    st.header("🟢 Unbiased FLAN-T5 (RAG Debiased Mode)")
    st.markdown("""
    This mode uses Retrieval-Augmented Generation (RAG) to provide 
    factual, unbiased responses. The system retrieves relevant anti-bias 
    knowledge from the knowledge base and uses it to guide FLAN-T5 to 
    produce neutral, evidence-based answers that correct harmful stereotypes.
    """)
    
    # Example query selector
    selected_example_unbiased = st.sidebar.selectbox(
        "Select an example query:",
        ["-- Select an example --"] + example_queries,
        key="unbiased_example"
    )
    
    # Query input
    query_unbiased = st.text_area(
        "Enter your query:",
        value=selected_example_unbiased if selected_example_unbiased != "-- Select an example --" else "",
        height=100,
        key="query_unbiased",
        placeholder="Type your question here..."
    )
    
    # Submit button
    if st.button("Generate Unbiased Response", type="primary", key="btn_unbiased"):
        if query_unbiased.strip():
            with st.spinner("Generating unbiased response (this may take a moment)..."):
                output = unbiased_mode(query_unbiased)
                st.subheader("Unbiased RAG Response:")
                st.text_area(
                    "Response:",
                    value=output,
                    height=300,
                    key="output_unbiased",
                    label_visibility="collapsed"
                )
        else:
            st.warning("Please enter a query.")

# Footer
st.markdown("---")
st.markdown("""
**Note:** All queries and responses are automatically logged to `web/logs/` for analysis.
- Biased mode logs: `web/logs/biased_logs.txt`
- Unbiased mode logs: `web/logs/unbiased_logs.txt`
""")
