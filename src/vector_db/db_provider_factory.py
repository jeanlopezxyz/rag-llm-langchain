# src/vector_db/db_provider_factory.py

# MODIFICACIÓN: Se eliminaron todas las importaciones innecesarias de 'llm', 
# 'langchain' y 'queue', que causaban la dependencia circular.

from vector_db.db_provider import DBProvider
from vector_db.faiss_provider import FAISSProvider
from vector_db.pgvector_provider import PGVectorProvider
from vector_db.redis_provider import RedisProvider
from vector_db.elastic_provider import ElasticProvider

# Constantes para los tipos de base de datos
PGVECTOR = "PGVECTOR"
REDIS = "REDIS"
FAISS = "FAISS"
ELASTIC = "ELASTIC"

class DBFactory:
    """
    Factory para crear y gestionar instancias de proveedores de bases de datos vectoriales.
    """
    def __init__(self):
        self.providers: dict[str, DBProvider] = {}

    def create_db_provider(self, db_type: str) -> DBProvider:
        """Crea una nueva instancia de un proveedor de base de datos."""
        if db_type == PGVECTOR:
            return PGVectorProvider()
        elif db_type == REDIS:
            return RedisProvider()
        elif db_type == FAISS:
            return FAISSProvider()
        elif db_type == ELASTIC:
            return ElasticProvider()
        else:
            raise ValueError(f"Proveedor de base de datos no soportado: {db_type}")

    def get_db_provider(self, db_type: str) -> DBProvider:
        """Obtiene una instancia de un proveedor, creándola si no existe."""
        if db_type not in self.providers:
            self.providers[db_type] = self.create_db_provider(db_type)
        
        return self.providers[db_type]
    
    def get_retriever(self, db_type: str):
        """Obtiene el retriever directamente de un proveedor de base de datos."""
        return self.get_db_provider(db_type).get_retriever()

    @classmethod 
    def get_providers(cls) -> list[str]:
        """Retorna una lista de los tipos de proveedores soportados."""
        return [PGVECTOR, REDIS, FAISS, ELASTIC]