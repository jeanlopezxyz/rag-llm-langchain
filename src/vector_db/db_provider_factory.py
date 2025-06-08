import os
from typing import Tuple
from llm.huggingface_provider import HuggingFaceProvider
from llm.llm_provider import LLMProvider
from llm.nemo_provider import NeMoProvider
from llm.openai_provider import OpenAIProvider
from langchain.llms.base import LLM
from queue import Queue

from vector_db.db_provider import DBProvider
from vector_db.faiss_provider import FAISSProvider
from vector_db.pgvector_provider import PGVectorProvider
from vector_db.redis_provider import RedisProvider
from vector_db.elastic_provider import ElasticProvider

PGVECTOR = "PGVECTOR"
REDIS = "REDIS"
FAISS = "FAISS"
ELASTIC = "ELASTIC"

class DBFactory:
    providers: dict[str, DBProvider] = {}
    def __init__(self):
        pass

    def create_db_provider(self, type: str) -> DBProvider:
        if type == PGVECTOR:
            return PGVectorProvider()
        elif type == REDIS:
            return RedisProvider()
        elif type == FAISS:
            return FAISSProvider()
        elif type == ELASTIC:
            return ElasticProvider()
        else:
            raise ValueError(type)

    def get_db_provider(self, type: str):
        if type not in self.providers:
            self.providers[type] = self.create_db_provider(type)
        
        return self.providers[type]
    
    def get_retriever(self, type: str):
        return self.get_db_provider(type).get_retriever()

    @classmethod 
    def get_providers(cls) -> list[str]:
        return [PGVECTOR, REDIS, FAISS, ELASTIC]
