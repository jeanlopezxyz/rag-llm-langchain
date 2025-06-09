# src/vector_db/db_provider.py

from typing import Optional
# 1. Cambia la importación a la nueva ruta
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

class DBProvider:
    """Base class for DB Provider.
    """
    embeddings: Optional[Embeddings] = None
    def __init__(self) -> None:
        # El resto del código no necesita cambios
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