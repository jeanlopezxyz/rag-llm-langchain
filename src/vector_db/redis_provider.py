from typing import Optional
# FIX: Actualizar imports deprecados de Redis
from langchain_community.vectorstores import Redis
from langchain_community.vectorstores.redis.base import RedisVectorStoreRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from vector_db.db_provider import DBProvider
import os

class RedisProvider(DBProvider):
    type = "REDIS"
    url: Optional[str] = None
    index: Optional[str] = None
    schema: Optional[str] = None
    retriever: Optional[VectorStoreRetriever] = None
    db: Optional[Redis] = None

    def __init__(self):
        super().__init__()
        self.url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.index = os.getenv('REDIS_INDEX', 'speaker_events')
        self.schema = os.getenv('REDIS_SCHEMA', 'redis_schema.yaml')
        
        # Validaciones mejoradas
        if not self.url:
            raise ValueError("REDIS_URL is not specified")
        if not self.index:
            raise ValueError("REDIS_INDEX is not specified")

    @classmethod
    def _get_type(cls) -> str:
        """Returns type of the db provider"""
        return cls.type
    
    def get_retriever(self) -> VectorStoreRetriever:
        if self.retriever is None:
            try:
                # FIX: Usar método más robusto para Redis
                self.db = Redis.from_existing_index(
                    embedding=self.get_embeddings(),  # FIX: usar 'embedding' en lugar de embedding function
                    redis_url=self.url,
                    index_name=self.index,
                    schema=self.schema
                )
                
                self.retriever = self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4, "score_threshold": 0.5}  # FIX: usar score_threshold
                )
            except Exception as e:
                print(f"❌ Error conectando a Redis: {e}")
                raise e
         
        return self.retriever