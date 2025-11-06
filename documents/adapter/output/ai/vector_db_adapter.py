# vector_db_adapter.py
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from documents.domain.port.vector_db_port import VectorDBPort


class FAISSVectorDBAdapter(VectorDBPort):
    def __init__(self):
        # 로컬 임베딩 모델
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 빈 FAISS Index 초기화 (MiniLM-L6-v2 = 384차원)
        embedding_dim = 384
        index = faiss.IndexFlatL2(embedding_dim)

        # 비어 있는 VectorStore 생성
        self.db = FAISS(embedding_function=self.embedding, index=index, docstore={}, index_to_docstore_id={})

    def add_document(self, doc_id: str, content: str):
        self.db.add_texts([content], ids=[doc_id])

    def search_similar(self, query: str, top_k: int = 5):
        return self.db.similarity_search(query, k=top_k)

    def as_retriever(self):
        return self.db.as_retriever()
