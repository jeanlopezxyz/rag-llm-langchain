# src/vector_db/db_provider.py

from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
import logging

logger = logging.getLogger(__name__)

class DBProvider:
    """Clase base para proveedores de bases de datos."""
    embeddings: Optional[Embeddings] = None
    
    def __init__(self) -> None:
        # Modelo de embedding unificado
        embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        
        # --- MODIFICACIÓN CLAVE ---
        # Nos aseguramos de que la normalización de embeddings esté activada.
        # Esto es crucial para que la búsqueda por similitud de coseno funcione correctamente.
        encode_kwargs = {'normalize_embeddings': True}
        
        logger.info(f"✅ Inicializando embeddings con el modelo: {embedding_model_name} (Normalización Activada)")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs=encode_kwargs
        )
        # --- FIN DE MODIFICACIÓN ---

    def get_retriever(self) -> VectorStoreRetriever:
        pass

    def get_embeddings(self) -> Embeddings:
        return self.embeddings