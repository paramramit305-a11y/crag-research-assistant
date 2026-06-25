import os
import re
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from rag_core import Embeddingmanager, VectorStoreManager, RAGRetriever

load_dotenv()

web_search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

class RAGState(TypedDict):
    query: str
    original_query: str
    documents: list
    answer: str
    retry_count: int
    is_relevant: str
    source: str


embedding_manager = Embeddingmanager()
vector_store = VectorStoreManager()
rag_retriever = RAGRetriever(embedding_manager, vector_store)

llm = ChatGroq(model="qwen/qwen3-32b", groq_api_key=os.getenv("GROQ_API_KEY"))


def clean_llm_output(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def retrieve_node(state: RAGState):
    print("--- RETRIEVE NODE ---")
    query = state["query"]
    documents = rag_retriever.retrieve(query, top_k=5)
    return {
        "documents": documents,
        "retry_count": state.get("retry_count", 0),
        "original_query": state.get("original_query", query)
    }


def grade_node(state: RAGState):
    print("--- GRADE NODE ---")
    query = state["query"]
    documents = state["documents"]

    if not documents:
        return {"is_relevant": "no"}

    relevant_docs = []
    irrelevant_docs = []

    for doc in documents:
        grading_prompt = f"""You are a relevance grader.
Does the following document contain useful information to answer the query?

Document:
{doc["document"]}

Query: {query}

Reply with ONLY 'yes' or 'no'."""

        response = llm.invoke(grading_prompt)
        grade = clean_llm_output(response.content).lower()

        if "yes" in grade:
            relevant_docs.append(doc)
        else:
            irrelevant_docs.append(doc)

    print(f"Relevant: {len(relevant_docs)} | Irrelevant: {len(irrelevant_docs)}")

    total = len(documents)
    relevant_count = len(relevant_docs)

    if relevant_count == 0:
        return {"is_relevant": "no", "documents": documents}
    elif relevant_count == total:
        return {"is_relevant": "yes", "documents": relevant_docs}
    else:
        return {"is_relevant": "ambiguous", "documents": relevant_docs}


def knowledge_refine_node(state: RAGState):
    print("--- KNOWLEDGE REFINE NODE ---")
    query = state["query"]
    documents = state["documents"]

    refined_docs = []

    for doc in documents:
        refine_prompt = f"""You are a knowledge extractor.
From the document below, extract ONLY the sentences or phrases that are directly useful for answering the query.
Remove all irrelevant, redundant, or noisy content.
Return only the refined knowledge as plain text.

Document:
{doc["document"]}

Query: {query}

Refined knowledge:"""

        response = llm.invoke(refine_prompt)
        refined_text = clean_llm_output(response.content).strip()

        if refined_text:
            refined_docs.append({
                **doc,
                "document": refined_text
            })

    print(f"Refined {len(refined_docs)} documents")
    return {"documents": refined_docs}


def retry_node(state: RAGState):
    print("--- RETRY NODE ---")
    original_query = state.get("original_query", state["query"])
    retry_count = state["retry_count"]

    retry_prompt = f"""The original query was: {original_query}
Rephrase this query differently to retrieve more relevant documents from a vector store.
Return ONLY the rephrased query, nothing else."""

    response = llm.invoke(retry_prompt)
    new_query = clean_llm_output(response.content).strip()

    print(f"Original: {original_query}")
    print(f"Rephrased: {new_query}")

    return {
        "query": new_query,
        "retry_count": retry_count + 1
    }


def web_search_node(state: RAGState):
    print("--- WEB SEARCH NODE ---")
    original_query = state.get("original_query", state["query"])

    search_prompt = f"""Convert this query into an effective web search query.
Return ONLY the search query, nothing else.

Query: {original_query}"""

    response = llm.invoke(search_prompt)
    search_query = clean_llm_output(response.content).strip()

    print(f"Web search query: {search_query}")

    results = web_search_tool.invoke(search_query)

    web_docs = []
    for result in results:
        web_docs.append({
            "id": f"web_{result.get('url', '')}",
            "document": result.get("content", ""),
            "metadata": {
                "source": result.get("url", ""),
                "type": "web_search"
            },
            "similarity_score": 0.5,
            "rank": len(web_docs) + 1
        })

    print(f"Web search returned {len(web_docs)} results")
    return {"documents": web_docs, "source": "web_search"}


def generate_node(state: RAGState):
    print("--- GENERATE NODE ---")
    original_query = state.get("original_query", state["query"])
    documents = state["documents"]

    context = "\n\n".join([doc["document"] for doc in documents])

    generation_prompt = f"""Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {original_query}

Answer:"""

    response = llm.invoke(generation_prompt)
    content = clean_llm_output(response.content)

    return {"answer": content}


def route_after_grading(state: RAGState) -> str:
    is_relevant = state["is_relevant"]
    retry_count = state["retry_count"]

    if is_relevant == "yes":
        return "refine"
    elif is_relevant == "ambiguous":
        if retry_count < 1:
            return "retry"
        else:
            return "refine"
    else:
        if retry_count < 2:
            return "retry"
        else:
            return "web_search"


graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.add_node("refine", knowledge_refine_node)
graph.add_node("retry", retry_node)
graph.add_node("generate", generate_node)
graph.add_node("web_search", web_search_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade")

graph.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "refine": "refine",
        "retry": "retry",
        "web_search": "web_search"
    }
)

graph.add_edge("refine", "generate")
graph.add_edge("retry", "retrieve")
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({
        "query": "What is attention mechanism in transformers?",
        "original_query": "",
        "documents": [],
        "answer": "",
        "retry_count": 0,
        "is_relevant": "",
        "source": "vector_store"
    })

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])
    print(f"\nSource: {result.get('source', 'vector_store')}")