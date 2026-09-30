import time
import streamlit as st
from agentic_rag import app
import re

def fix_latex(text: str) -> str:
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text

st.set_page_config(page_title="CRAG Research Assistant", page_icon="🧠", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at 20% 0%, #1a1033 0%, #0b0e17 45%, #0b0e17 100%);
}
.crag-hero {
    padding: 2.2rem 2.5rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(37,99,235,0.10));
    border: 1px solid rgba(148,110,255,0.25);
    margin-bottom: 1.6rem;
}
.crag-hero h1 {
    font-size: 2.1rem;
    margin: 0;
    background: linear-gradient(90deg, #b794f6, #63b3ed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.crag-hero p {
    color: #a8b0c3;
    margin-top: 0.4rem;
    font-size: 0.95rem;
}
.crag-badges span {
    display: inline-block;
    background: rgba(148,110,255,0.12);
    border: 1px solid rgba(148,110,255,0.3);
    color: #c9b8ff;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    margin-right: 6px;
    margin-top: 10px;
}
.crag-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.crag-source-vector {
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.35);
    color: #6ee7b7;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    display: inline-block;
}
.crag-source-web {
    background: rgba(96,165,250,0.12);
    border: 1px solid rgba(96,165,250,0.35);
    color: #93c5fd;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    display: inline-block;
}
.crag-metrics {
    display: flex;
    gap: 1.2rem;
    margin-top: 0.8rem;
    color: #8a92a6;
    font-size: 0.82rem;
}
div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.6rem;
    font-weight: 600;
}
div.stButton > button:hover {
    opacity: 0.9;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        "1. Retrieve top documents from local vector store\n"
        "2. Grade each document for relevance\n"
        "3. Refine or rephrase query on weak matches\n"
        "4. Fall back to live web search if needed\n"
        "5. Generate grounded answer"
    )
    st.divider()
    st.markdown("### Stack")
    st.markdown("LangGraph · LangChain · Groq (GPT-OSS) · Tavily · ChromaDB")
    st.divider()
    if st.session_state.history:
        st.markdown("### Recent Questions")
        for item in reversed(st.session_state.history[-5:]):
            st.caption(item["query"])

st.markdown(
    """
    <div class="crag-hero">
        <h1>CRAG — Corrective RAG Research Assistant</h1>
        <p>A self-correcting research pipeline: grades retrieved documents, refines knowledge, and falls back to live web search when local papers fall short.</p>
        <div class="crag-badges">
            <span>LangGraph</span>
            <span>Per-document Grading</span>
            <span>Knowledge Refinement</span>
            <span>Web Search Fallback</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Ask a question",
        placeholder="e.g. What is the attention mechanism in transformers?",
        label_visibility="collapsed",
    )
with col2:
    run_search = st.button("Search", use_container_width=True)

if run_search and query.strip():
    with st.spinner("Running CRAG pipeline..."):
        start_time = time.time()
        try:
            result = app.invoke(
                {
                    "query": query,
                    "original_query": query,
                    "documents": [],
                    "answer": "",
                    "retry_count": 0,
                    "is_relevant": "",
                    "source": "vector_store",
                }
            )
        except Exception as e:
            st.error("Something went wrong while running the pipeline. Please retry in a moment.")
            with st.expander("Debug details"):
                st.exception(e)
            st.stop()
        elapsed = time.time() - start_time

    st.session_state.history.append({"query": query})

    source = result.get("source", "vector_store")
    retry_count = result.get("retry_count", 0)
    source_label = "Web Search" if source == "web_search" else "Local Vector Store"
    badge_class = "crag-source-web" if source == "web_search" else "crag-source-vector"

    st.markdown(
        f"""
        <div class="crag-card">
            <span class="{badge_class}">Source: {source_label}</span>
            <div class="crag-metrics">
                <span>⏱ {elapsed:.1f}s</span>
                <span>🔁 {retry_count} retry attempt(s)</span>
            </div>
            <h4 style="margin-top:1.2rem;">Answer</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(fix_latex(result["answer"]))
    
    docs = result.get("documents", [])
    if docs:
        with st.expander("Sources"):
            seen = set()
            for d in docs:
                src = d.get("metadata", {}).get("source", "unknown")
                if src not in seen:
                    seen.add(src)
                    st.write(f"- {src}")
    
elif run_search:
    st.warning("Please enter a question first.")
