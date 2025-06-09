from typing import Optional
# FIX: Actualizar import deprecado de HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

class DBProvider:
    """Base class for DB Provider.
    """
    embeddings: Optional[Embeddings] = None
    def __init__(self) -> None:
        # FIX: Especificar modelo explícitamente para evitar warning
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        pass

    def _get_type(self) -> str:
        pass

    def get_retriever(self) -> VectorStoreRetriever:
        pass

    def get_embeddings(self) -> Embeddings:
        return self.embeddings