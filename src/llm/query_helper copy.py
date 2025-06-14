# src/llm/query_helper.py
import os
import logging
import re
from typing import List, Optional, Any, Dict, Set
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers.cross_encoder import CrossEncoder
from vector_db.db_provider_factory import DBFactory
from pydantic import Field, ConfigDict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

# Configuración de logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVENT_NAME = os.getenv("EVENT_NAME", "KCD Guatemala 2025")

# Prompt genérico y restrictivo
GENERIC_PROMPT = """Eres un asistente para responder preguntas sobre eventos.

REGLAS ESTRICTAS:
- Responde SOLO con la información proporcionada
- NO inventes ni agregues información
- Sé conciso y directo
- Si no tienes la información, di "No tengo esa información"

Contexto:
{context}

Pregunta: {input}

Respuesta:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(GENERIC_PROMPT)


class EventKnowledgeBase:
    """Base de conocimiento genérica para eventos"""
    
    def __init__(self, all_docs: List[Document], embeddings: Any):
        logger.info(f"Inicializando base de conocimiento con {len(all_docs)} documentos")
        self.documents = self._process_documents(all_docs)
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        self._build_indices()
    
    def _process_documents(self, all_docs: List[Document]) -> List[Document]:
        """Procesa documentos sin asumir estructura específica"""
        return [doc for doc in all_docs if doc.metadata.get("source")]
    
    def _build_indices(self):
        """Construye índices genéricos basados en metadata disponible"""
        self.indices = {
            'by_speaker': {},
            'by_session': {},
            'by_metadata': {}
        }
        
        for doc in self.documents:
            metadata = doc.metadata
            
            # Índice por speaker (si existe)
            if 'speaker_names' in metadata:
                speakers = metadata['speaker_names'].split(',')
                for speaker in speakers:
                    speaker = speaker.strip()
                    if speaker:
                        if speaker not in self.indices['by_speaker']:
                            self.indices['by_speaker'][speaker] = []
                        self.indices['by_speaker'][speaker].append(doc)
            
            # Índice por sesión (si existe)
            if 'session_name' in metadata:
                session = metadata['session_name']
                self.indices['by_session'][session] = doc
            
            # Índice por otros metadatos
            for key, value in metadata.items():
                if key not in self.indices['by_metadata']:
                    self.indices['by_metadata'][key] = set()
                self.indices['by_metadata'][key].add(str(value))


class GenericEventRetriever(BaseRetriever):
    """Retriever genérico para eventos"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    knowledge_base: EventKnowledgeBase = Field(...)
    cross_encoder: Optional[CrossEncoder] = Field(default=None)
    
    def __init__(self, knowledge_base: EventKnowledgeBase, **kwargs):
        super().__init__(knowledge_base=knowledge_base, **kwargs)
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except:
            logger.warning("CrossEncoder no disponible")
            self.cross_encoder = None
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extrae entidades de la consulta de manera genérica"""
        entities = {
            'names': [],
            'topics': [],
            'times': [],
            'query_type': 'general'
        }
        
        query_lower = query.lower()
        
        # Detectar tipo de consulta
        if any(word in query_lower for word in ['cuántas', 'cuantas', 'total', 'cantidad']):
            entities['query_type'] = 'count'
        elif any(word in query_lower for word in ['cuándo', 'cuando', 'fecha', 'día']):
            entities['query_type'] = 'when'
        elif any(word in query_lower for word in ['dónde', 'donde', 'lugar', 'ubicación']):
            entities['query_type'] = 'where'
        elif any(word in query_lower for word in ['charlas sobre', 'charlas de', 'sesiones sobre', 'talks about']):
            entities['query_type'] = 'topic_search'
        
        # Extraer nombres propios (palabras capitalizadas)
        words = query.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Verificar si es parte de un nombre completo
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    entities['names'].append(f"{word} {words[i + 1]}")
                elif word not in ['El', 'La', 'Los', 'Las', 'De', 'Del']:
                    entities['names'].append(word)
        
        # Extraer temas (palabras después de "sobre", "de", etc.)
        topic_patterns = [
            r'(?:charlas?|sesion(?:es)?|talks?)\s+(?:sobre|de|about)\s+(\w+(?:\s+\w+)*)',
            r'(?:tema|topic)\s+(?:de|sobre|about)\s+(\w+(?:\s+\w+)*)'
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, query_lower)
            entities['topics'].extend(matches)
        
        # Extraer referencias temporales
        if 'mañana' in query_lower:
            entities['times'].append('morning')
        if 'tarde' in query_lower:
            entities['times'].append('afternoon')
        
        return entities
    
    def _search_by_similarity(self, query: str, k: int = 20) -> List[Document]:
        """Búsqueda por similitud semántica"""
        return self.knowledge_base.vector_store.similarity_search(query, k=k)
    
    def _filter_by_metadata(self, docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """Filtra documentos por metadata de manera genérica"""
        filtered = []
        
        for doc in docs:
            matches = True
            for key, value in filters.items():
                doc_value = doc.metadata.get(key, '')
                if isinstance(value, list):
                    # Si el filtro es una lista, verificar si algún valor está presente
                    if not any(v.lower() in str(doc_value).lower() for v in value):
                        matches = False
                        break
                else:
                    # Comparación simple
                    if value.lower() not in str(doc_value).lower():
                        matches = False
                        break
            
            if matches:
                filtered.append(doc)
        
        return filtered
    
    def _search_by_topic(self, topics: List[str]) -> List[Document]:
        """Búsqueda genérica por tema"""
        all_results = []
        seen_ids = set()
        
        for topic in topics:
            # Búsqueda semántica por tema
            docs = self._search_by_similarity(f"{topic} charlas sesiones talks {topic}", k=30)
            
            for doc in docs:
                # Verificar relevancia del tema en múltiples campos
                content_lower = doc.page_content.lower()
                metadata_str = ' '.join(str(v) for v in doc.metadata.values()).lower()
                
                if topic.lower() in content_lower or topic.lower() in metadata_str:
                    doc_id = id(doc)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_results.append(doc)
        
        return all_results
    
    def _search_by_speaker(self, names: List[str]) -> List[Document]:
        """Búsqueda genérica por nombre"""
        results = []
        seen_sessions = set()
        
        # Primero buscar en índice si existe
        for name in names:
            # Búsqueda exacta en índice
            if name in self.knowledge_base.indices['by_speaker']:
                for doc in self.knowledge_base.indices['by_speaker'][name]:
                    session = doc.metadata.get('session_name', '')
                    if session and session not in seen_sessions:
                        seen_sessions.add(session)
                        results.append(doc)
            
            # Búsqueda parcial en índice
            name_lower = name.lower()
            for speaker, docs in self.knowledge_base.indices['by_speaker'].items():
                if name_lower in speaker.lower():
                    for doc in docs:
                        session = doc.metadata.get('session_name', '')
                        if session and session not in seen_sessions:
                            seen_sessions.add(session)
                            results.append(doc)
        
        # Si no hay resultados en índice, búsqueda semántica
        if not results:
            for name in names:
                docs = self._search_by_similarity(f"speaker {name} ponente {name} presenter {name}", k=10)
                for doc in docs:
                    # Verificar que el nombre esté en la metadata relevante
                    if any(name.lower() in str(v).lower() for v in doc.metadata.values()):
                        session = doc.metadata.get('session_name', '')
                        if session and session not in seen_sessions:
                            seen_sessions.add(session)
                            results.append(doc)
        
        return results
    
    def _handle_count_query(self, query: str, entities: Dict) -> List[Document]:
        """Maneja consultas de conteo de manera genérica"""
        query_lower = query.lower()
        
        # Determinar qué contar
        if 'charla' in query_lower or 'sesion' in query_lower or 'session' in query_lower:
            # Contar sesiones únicas
            unique_sessions = set()
            for doc in self.knowledge_base.documents:
                if doc.metadata.get('session_name'):
                    unique_sessions.add(doc.metadata['session_name'])
            
            count = len(unique_sessions)
            return [Document(
                page_content=f"{count} charlas/sesiones.",
                metadata={"source": "count"}
            )]
        
        elif 'ponente' in query_lower or 'speaker' in query_lower:
            # Contar speakers únicos
            count = len(self.knowledge_base.indices['by_speaker'])
            return [Document(
                page_content=f"{count} ponentes/speakers.",
                metadata={"source": "count"}
            )]
        
        # Si no se puede determinar qué contar, devolver vacío
        return []
    
    def _handle_event_info_query(self, query_type: str) -> List[Document]:
        """Maneja consultas sobre información del evento"""
        # Buscar documentos que contengan información general del evento
        info_docs = []
        
        for doc in self.knowledge_base.documents:
            content_lower = doc.page_content.lower()
            
            if query_type == 'when' and any(word in content_lower for word in ['fecha', 'junio', 'sábado']):
                info_docs.append(doc)
            elif query_type == 'where' and any(word in content_lower for word in ['centro de convenciones', 'antigua', 'ubicación']):
                info_docs.append(doc)
        
        return info_docs[:1] if info_docs else []
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Obtiene documentos relevantes de manera genérica"""
        
        # Extraer entidades de la consulta
        entities = self._extract_entities(query)
        logger.info(f"Entidades extraídas: {entities}")
        
        # Manejar diferentes tipos de consultas
        if entities['query_type'] == 'count':
            return self._handle_count_query(query, entities)
        
        elif entities['query_type'] in ['when', 'where']:
            docs = self._handle_event_info_query(entities['query_type'])
            if docs:
                return docs
        
        elif entities['query_type'] == 'topic_search' and entities['topics']:
            return self._search_by_topic(entities['topics'])[:10]
        
        elif entities['names']:
            return self._search_by_speaker(entities['names'])[:10]
        
        # Búsqueda general por similitud
        docs = self._search_by_similarity(query, k=20)
        
        # Aplicar filtros si hay entidades específicas
        if entities['times']:
            time_filtered = []
            for doc in docs:
                start_time = doc.metadata.get('start_time', '')
                if start_time:
                    try:
                        hour = int(start_time.split(':')[0])
                        if 'morning' in entities['times'] and hour < 12:
                            time_filtered.append(doc)
                        elif 'afternoon' in entities['times'] and hour >= 12:
                            time_filtered.append(doc)
                    except:
                        pass
            docs = time_filtered if time_filtered else docs
        
        # Re-ranking si está disponible
        if self.cross_encoder and len(docs) > 1:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            docs = [doc for _, doc in ranked_docs]
        
        return docs[:10]


def format_docs_for_llm(docs: List[Document]) -> str:
    """Formatea documentos de manera genérica"""
    if not docs:
        return "No encontré información sobre esa consulta."
    
    # Para resultados simples (count, info)
    if len(docs) == 1 and docs[0].metadata.get("source") in ["count", "event_info"]:
        return docs[0].page_content
    
    # Formatear múltiples resultados
    formatted_parts = []
    seen_items = set()
    
    for doc in docs:
        # Intentar extraer información clave de manera genérica
        key_info = []
        
        # Título/Nombre de sesión
        for field in ['session_name', 'title', 'name']:
            if doc.metadata.get(field):
                key_info.append(doc.metadata[field])
                break
        
        # Speaker/Ponente
        for field in ['speaker_names', 'speaker', 'presenter']:
            if doc.metadata.get(field):
                key_info.append(f"- {doc.metadata[field]}")
                break
        
        # Tiempo
        for field in ['start_time', 'time', 'schedule']:
            if doc.metadata.get(field):
                key_info.append(f"({doc.metadata[field]})")
                break
        
        if key_info:
            info_str = " ".join(key_info)
            if info_str not in seen_items:
                seen_items.add(info_str)
                formatted_parts.append(f"• {info_str}")
    
    if formatted_parts:
        if len(formatted_parts) == 1:
            return formatted_parts[0].replace("• ", "")
        else:
            return f"Encontré {len(formatted_parts)} resultados:\n" + "\n".join(formatted_parts)
    
    # Si no se pudo formatear, extraer del contenido
    return docs[0].page_content[:500] if docs else "No encontré información."


# Variables globales
KNOWLEDGE_BASE: Optional[EventKnowledgeBase] = None


def initialize_knowledge_base():
    """Inicializa la base de conocimiento"""
    global KNOWLEDGE_BASE
    if KNOWLEDGE_BASE is None:
        logger.info("Inicializando base de conocimiento...")
        try:
            db_type = os.getenv("DB_TYPE", "PGVECTOR")
            db_factory = DBFactory()
            base_retriever = db_factory.get_retriever(db_type)
            embeddings = base_retriever.vectorstore.embeddings
            
            all_docs = base_retriever.vectorstore.similarity_search(query=" ", k=2000)
            logger.info(f"Recuperados {len(all_docs)} documentos")
            
            KNOWLEDGE_BASE = EventKnowledgeBase(all_docs, embeddings)
        except Exception as e:
            logger.error(f"Error inicializando: {e}")
            raise
    return KNOWLEDGE_BASE


def create_qa_chain(knowledge_base: EventKnowledgeBase, llm: Any) -> Runnable:
    """Crea la cadena QA genérica"""
    retriever = GenericEventRetriever(knowledge_base=knowledge_base)
    
    chain = (
        RunnableParallel(
            context=(lambda x: x['input']) | retriever | format_docs_for_llm,
            input=(lambda x: x['input']),
            event_name=lambda x: EVENT_NAME
        )
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return chain