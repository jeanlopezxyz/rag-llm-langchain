from typing import Optional
from vector_db.db_provider import DBProvider
# FIX: Actualizar import deprecado de PGVector
from langchain_community.vectorstores import PGVector
from langchain_core.vectorstores import VectorStoreRetriever
import os

class PGVectorProvider(DBProvider):
    type = "PGVECTOR"
    url: Optional[str] = None
    collection_name: Optional[str] = None
    retriever: Optional[VectorStoreRetriever] = None
    db: Optional[PGVector] = None
    
    def __init__(self):
        super().__init__()
        # Configurar variables de entorno con valores por defecto más robustos
        self.host = os.getenv('PGVECTOR_HOST', 'localhost')
        self.port = os.getenv('PGVECTOR_PORT', '5432')
        self.database = os.getenv('PGVECTOR_DATABASE', 'vectordb')
        self.user = os.getenv('PGVECTOR_USER', 'postgres')
        self.password = os.getenv('PGVECTOR_PASSWORD', 'password')
        
        # Construir URL de conexión
        self.url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        self.collection_name = os.getenv('PGVECTOR_COLLECTION_NAME', 'embeddings')
        
        # Validaciones mejoradas
        if not self.host:
            raise ValueError("PGVECTOR_HOST is not specified")
        if not self.password or self.password == 'password':
            print("⚠️ Warning: Using default password for PostgreSQL")

    @classmethod
    def _get_type(cls) -> str:
        """Returns type of the db provider"""
        return cls.type
    
    def get_retriever(self) -> VectorStoreRetriever:
        if self.retriever is None:
            try:
                # FIX: Usar método más robusto para crear PGVector
                self.db = PGVector(
                    connection_string=self.url,
                    collection_name=self.collection_name,
                    embedding_function=self.get_embeddings(),
                    # FIX: Agregar configuración adicional para evitar warnings
                    pre_delete_collection=False,
                    logger=None
                )

                self.retriever = self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4, "score_threshold": 0.5}  # FIX: usar score_threshold
                )
            except Exception as e:
                print(f"❌ Error conectando a PostgreSQL: {e}")
                raise e
         
        return self.retriever