import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Embeddingmanager():
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print("loading model....", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        print("embedding dimensions=", self.model.get_embedding_dimension())

    def generate_embeddings(self, text):
        embeddings = self.model.encode(text, show_progress_bar=True)
        print("embeddings shape:", embeddings.shape)
        return embeddings


class VectorStoreManager:
    def __init__(self, persist_directory="data/vector_store", collection_name="pdf_documents"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "vector store collection for pdf embeddings in RAG"}
        )
        print("initialized the vector store with collection:", self.collection_name)
        print("docs in collection:", self.collection.count())

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("num of documents does not match num of embeddings")

        ids = []
        all_metadata = []
        documents_content = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            all_metadata.append(metadata)
            documents_content.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documents_content,
            embeddings=embeddings_list
        )
        print("total documents added in vector store =", len(documents_content))
        print("docs in collection:", self.collection.count())


class RAGRetriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        query_embeddings = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embeddings.tolist()],
            n_results=top_k,
        )

        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for i, (doc_id, metadata, document, distance) in enumerate(zip(ids, metadatas, documents, distances)):
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "document": document,
                        "metadata": metadata,
                        "distance": distance,
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    })
            print(f"retrieved {len(retrieved_docs)} documents")
        else:
            print("no documents found")

        return retrieved_docs