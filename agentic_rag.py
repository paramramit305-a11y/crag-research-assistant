import os
import re
import time
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from groq import RateLimitError, APIStatusError
from rag_core import Embeddingmanager, VectorStoreManager, RAGRetriever

load_dotenv()

web_search_tool = TavilySearch(
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

generation_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    reasoning_effort="medium"
)

utility_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    reasoning_effort="low"
)


def safe_llm_invoke(llm_instance, prompt, max_retries=4, base_wait=4):
    for attempt in range(max_retries):
        try:
            return llm_instance.invoke(prompt)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = base_wait * (2 ** attempt)
            print(f"Rate limited — retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except APIStatusError as e:
            print(f"Groq API error (not a rate limit): {e}")
            raise


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

    docs_block = "\n\n".join(
        [f"[Document {i + 1}]\n{doc['document']}" for i, doc in enumerate(documents)]
    )

    grading_prompt = f"""You are a strict relevance grader.

For each document below, decide if it DIRECTLY and specifically helps answer the query.
Sharing a topic, keyword, or general subject area is NOT enough - the document must actually
help answer this exact question with concrete relevant information.

{docs_block}

Query: {query}

Reply with exactly {len(documents)} lines, one per document, in this exact format:
1: yes
2: no
3: yes

No other text, no explanation."""

    response = safe_llm_invoke(utility_llm, grading_prompt)
    grades_text = clean_llm_output(response.content)

    grades = {}
    for line in grades_text.splitlines():
        match = re.match(r"\s*(\d+)\s*[:\-]\s*(yes|no)", line.strip(), re.IGNORECASE)
        if match:
            grades[int(match.group(1))] = match.group(2).lower()

    relevant_docs = []
    for i, doc in enumerate(documents):
        grade = grades.get(i + 1, "no")
        if grade == "yes":
            relevant_docs.append(doc)

    print(f"Relevant: {len(relevant_docs)} | Irrelevant: {len(documents) - len(relevant_docs)}")

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

        response = safe_llm_invoke(utility_llm, refine_prompt)
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

    response = safe_llm_invoke(utility_llm, retry_prompt)
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

    response = safe_llm_invoke(utility_llm, search_prompt)
    search_query = clean_llm_output(response.content).strip()

    print(f"Web search query: {search_query}")

    raw_results = web_search_tool.invoke(search_query)
    result_list = raw_results.get("results", []) if isinstance(raw_results, dict) else raw_results

    web_docs = []
    for result in result_list:
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

    response = safe_llm_invoke(generation_llm, generation_prompt)
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
        if retry_count < 1:
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
