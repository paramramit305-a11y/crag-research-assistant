# CRAG — Corrective RAG Research Assistant

A self-correcting RAG system built with LangGraph — grades each retrieved document individually, refines relevant knowledge, and falls back to live web search when local documents are insufficient. Inspired by the CRAG paper (arXiv:2401.15884).

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-state_machine-1C3C3C?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-pipeline-1C3C3C?style=flat-square)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-qwen3--32b-F55036?style=flat-square)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)

**[Live demo →](https://crag-research-assistant.streamlit.app)**

## What it does

Ask a question about AI/ML research — the system searches a local vector store of 9 research papers, grades each result, and decides what to do next:

- If documents are relevant — refines them to extract only the useful parts, then generates an answer
- If documents are partially relevant — retries with a rephrased query before deciding
- If local documents consistently fail — falls back to a live Tavily web search and answers from that instead
- Every response tells you whether the answer came from local papers or the web

## Demo

Local paper query — answer found in vector store:

<img width="1366" height="626" alt="Screenshot 2026-06-25 123925" src="https://github.com/user-attachments/assets/51b8bbdd-4040-4b0a-902f-60544ac9b9f5" />

When a query falls outside the knowledge base, the system routes to web search automatically and flags it in the UI instead of answering from irrelevant chunks.

## Why I built it this way

Standard RAG pipelines have a silent failure mode: the retriever always returns *something*, and the LLM generates an answer regardless of whether that something is actually relevant. The result is confident-sounding hallucinations with no indication that retrieval went wrong.

I wanted to fix that. After building a basic RAG pipeline (LangChain + ChromaDB) as an earlier project, I came across the CRAG paper (arXiv:2401.15884) which formalises exactly this problem — retrieved documents need to be evaluated before generation, not blindly trusted. I used LangGraph to implement this as an explicit state machine, so every routing decision (retry, refine, web search) is a visible node in the graph rather than hidden inside prompt logic.

The per-document grading was a deliberate choice over grading the full context at once — grading all retrieved chunks together means one relevant document can mask four irrelevant ones. Grading individually and filtering before generation gives the LLM a cleaner context to work with.

## How it works

```
app.py (Streamlit)
   │  takes a query, calls LangGraph app.invoke()
   ▼
agentic_rag.py (LangGraph StateGraph)
   │
   ├── retrieve_node      → ChromaDB semantic search (top-5 chunks)
   │
   ├── grade_node         → LLM grades each document individually (yes/no)
   │                        returns: "yes" / "no" / "ambiguous"
   │
   ├── [conditional edge] → yes      → knowledge_refine_node
   │                        ambiguous → retry_node (if retry_count < 1)
   │                                    else knowledge_refine_node
   │                        no        → retry_node (if retry_count < 2)
   │                                    else web_search_node
   │
   ├── knowledge_refine_node → LLM strips irrelevant content from each doc
   │
   ├── retry_node         → LLM rephrases query, loops back to retrieve
   │
   ├── web_search_node    → LLM generates optimised search query → Tavily
   │
   └── generate_node      → answers from original query against refined context
```

The state (`RAGState`) carries `original_query` separately from `query` so that after retries and rephrasing, the final answer is always generated against what the user actually asked — not a mid-pipeline reformulation.

## Stack

| Part | Choice | Why |
|---|---|---|
| Graph orchestration | LangGraph | Explicit state machine — routing decisions are nodes, not hidden prompt logic |
| Vector store | ChromaDB (persistent) | Local, no API cost, persists across runs |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Runs fully on CPU, zero API cost |
| LLM | Groq — `qwen/qwen3-32b` | Free tier, fast enough for multi-node graphs with several LLM calls per query |
| Web search fallback | Tavily | Clean API, returns structured content rather than raw HTML |
| Frontend | Streamlit | Fastest path from a working Python graph to a usable UI |

## Project structure

```
crag-research-assistant/
├── app.py               # Streamlit UI — takes query, calls graph, shows answer + source
├── agentic_rag.py       # LangGraph graph — all nodes, edges, routing logic
├── rag_core.py          # EmbeddingManager, VectorStoreManager, RAGRetriever classes
├── data/
│   ├── pdfs/            # 9 AI/ML research papers (ingested at setup)
│   └── vector_store/    # ChromaDB persistent store (2182 chunks)
└── requirements.txt
```

## Knowledge base

The vector store contains 9 AI/ML research papers — foundational and recent:

- Attention Is All You Need (Transformer architecture)
- RAG Survey (retrieval-augmented generation overview)
- BERT, GPT-3 (language model foundations)
- CRAG paper (arXiv:2401.15884 — what this system implements)
- Self-RAG, ReAct, Chain-of-Thought, LoRA

Questions outside this scope trigger the web search fallback automatically.

## Running it locally

```bash
git clone https://github.com/paramramit305-a11y/crag-research-assistant.git
cd crag-research-assistant
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

The vector store is already included in the repo — no ingestion step needed locally. Run directly:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## What this is and isn't

This is an implementation of the CRAG pattern (arXiv:2401.15884) built from scratch using LangGraph, not a wrapper around an existing CRAG library. The original paper uses a fine-tuned T5-large as the retrieval evaluator — this implementation uses an LLM-based grader via Groq instead, which trades evaluation speed for the ability to run without a separately fine-tuned model.

The knowledge base is intentionally scoped to AI/ML research papers. When a query falls outside that scope, the system says so explicitly via the web search fallback — it does not hallucinate an answer from unrelated retrieved chunks.

## Known limitations

- **Grading adds latency** — each retrieved document is graded with a separate LLM call, so a 5-document retrieval triggers 5 grading calls before generation. This is intentional (per-document precision over bulk grading) but makes the pipeline slower than standard RAG for simple queries.
- **Knowledge base is static** — adding new papers requires re-running the ingestion notebook locally and pushing the updated vector store to the repo.
- **Web search fallback depends on Tavily quota** — free tier has monthly request limits, so heavy usage will hit the cap.

## Possible next steps

- Add a confidence score to the source attribution — not just "local" or "web" but how many documents passed grading and what their similarity scores were
- Let users upload their own PDFs through the Streamlit UI and ingest on the fly, rather than requiring a fixed knowledge base
- Replace LLM-based grading with a fine-tuned evaluator model (closer to the original paper's T5-large approach) to reduce latency

## Author

Amit Parmar — BSc IT (AIML), Gokul Global University  
[GitHub](https://github.com/paramramit305-a11y) · [LinkedIn](https://www.linkedin.com/in/parmar-amit-97941a377) · [HuggingFace](https://huggingface.co/parmar-amit)
