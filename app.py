import streamlit as st
from agentic_rag import app

st.set_page_config(page_title="CRAG Research Assistant", layout="wide")

st.title("CRAG — Corrective RAG Research Assistant")
st.caption("LangGraph · Per-document grading · Knowledge refinement · Web search fallback")

st.divider()

query = st.text_input("Ask a question", placeholder="e.g. What is the attention mechanism in transformers?")

if st.button("Search") and query.strip():
    with st.spinner("Running CRAG pipeline..."):
        try:
            result = app.invoke({
                "query": query,
                "original_query": query,
                "documents": [],
                "answer": "",
                "retry_count": 0,
                "is_relevant": "",
                "source": "vector_store"
            })
        except Exception:
            st.error("Server thoda busy hai (rate limit) — kripya 30-60 seconds baad dobara try karo 🙏")
            st.stop()

    source = result.get("source", "vector_store")

    if source == "web_search":
        st.info("Local documents insufficient — answer retrieved from web search.")
    else:
        st.success("Answer found in local research papers.")

    st.subheader("Answer")
    st.write(result["answer"])
