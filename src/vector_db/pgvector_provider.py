# src/vector_db/pgvector_provider.py

from typing import Optional
from vector_db.db_provider import DBProvider
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever
import os

class PGVectorProvider(DBProvider):
    type = "PGVECTOR"
    url: Optional[str] = None
    collection_name: Optional[str] = None
    # Atributos inicializados como None para evitar errores
    retriever: Optional[VectorStoreRetriever] = None 
    db: Optional[PGVector] = None
    
    def __init__(self):
        super().__init__()
        self.host = os.getenv('PGVECTOR_HOST', 'localhost')
        self.port = os.getenv('PGVECTOR_PORT', '5432')
        self.database = os.getenv('PGVECTOR_DATABASE', 'vectordb')
        self.user = os.getenv('PGVECTOR_USER', 'postgres')
        self.password = os.getenv('PGVECTOR_PASSWORD', 'password')
        self.url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        self.collection_name = os.getenv('PGVECTOR_COLLECTION_NAME', 'embeddings')
        if not self.host:
            raise ValueError("PGVECTOR_HOST is not specified")
        if not self.password or self.password == 'password':
            print("⚠️ Warning: Using default password for PostgreSQL")

    @classmethod
    def _get_type(cls) -> str:
        return cls.type
    
    # Tu método get_retriever que ya está correcto
    def get_retriever(self) -> VectorStoreRetriever:
        if self.retriever is None:
            try:
                self.db = PGVector.from_existing_index(
                    connection_string=self.url,
                    collection_name=self.collection_name,
                    embedding=self.get_embeddings(),
                )

                self.retriever = self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
            except Exception as e:
                print(f"❌ Error conectando a PostgreSQL con langchain-postgres: {e}")
                raise e
         
        return self.retriever