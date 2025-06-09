# src/vector_db/pgvector_provider.py

import os
from typing import Optional
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever
from vector_db.db_provider import DBProvider
import logging

logger = logging.getLogger(__name__)

class PGVectorProvider(DBProvider):
    """
    Proveedor estándar de PGVector que se conecta a las tablas gestionadas por LangChain.
    """
    type = "PGVECTOR"
    url: Optional[str] = None
    collection_name: Optional[str] = None
    retriever: Optional[VectorStoreRetriever] = None
    db: Optional[PGVector] = None
    
    def __init__(self):
        super().__init__()
        # Construimos la URL a partir de las variables de entorno
        host = os.getenv('PGVECTOR_HOST', 'localhost')
        port = os.getenv('PGVECTOR_PORT', '5432')
        database = os.getenv('PGVECTOR_DATABASE', 'vectordb')
        user = os.getenv('PGVECTOR_USER', 'postgres')
        password = os.getenv('PGVECTOR_PASSWORD', 'password')
        self.url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

        self.collection_name = os.getenv('PGVECTOR_COLLECTION_NAME', 'embeddings')
        
        if not all([host, database, user, password]):
            raise ValueError("Faltan una o más variables de entorno de PGVECTOR (HOST, DATABASE, USER, PASSWORD)")
        
        logger.info(f"✅ PGVectorProvider (Estándar) inicializado para la colección '{self.collection_name}'.")

    @classmethod
    def _get_type(cls) -> str:
        """Returns type of the db provider"""
        return cls.type
    
    def get_retriever(self) -> VectorStoreRetriever:
        """
        Crea un recuperador estándar de LangChain que lee de las tablas
        'langchain_pg_collection' y 'langchain_pg_embedding'.
        """
        if self.retriever is None:
            self.db = PGVector(
                connection=self.url,
                collection_name=self.collection_name,
                embeddings=self.get_embeddings()
            )

            self.retriever = self.db.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 4})
         
        return self.retriever