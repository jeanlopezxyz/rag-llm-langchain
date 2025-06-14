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

# PROMPT COMPLETAMENTE LIMPIO - Sin estructura visible
IMPROVED_PROMPT = """Responde directamente la pregunta usando solo la información del contexto.

Contexto: {context}

{input}

"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(IMPROVED_PROMPT)


class EnhancedEventKnowledgeBase:
    """Base de conocimiento optimizada para la nueva estructura de base de datos"""
    
    def __init__(self, all_docs: List[Document], embeddings: Any):
        logger.info(f"Inicializando base de conocimiento optimizada con {len(all_docs)} documentos")
        self.documents = self._process_documents(all_docs)
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        self._build_enhanced_indices()
    
    def _process_documents(self, all_docs: List[Document]) -> List[Document]:
        """Procesa documentos optimizando la metadata para las nuevas estructuras"""
        processed = []
        for doc in all_docs:
            if doc.metadata.get("source"):
                # Enriquecer metadata con información temporal mejorada
                self._enrich_temporal_metadata(doc)
                # Enriquecer con información de speakers
                self._enrich_speaker_metadata(doc)
                # Enriquecer con información de ubicación
                self._enrich_location_metadata(doc)
                processed.append(doc)
        return processed
    
    def _enrich_temporal_metadata(self, doc: Document):
        """Enriquece la metadata temporal con mayor precisión"""
        start_time = doc.metadata.get('start_time', '')
        if start_time:
            try:
                # Extraer hora numérica
                hour = int(start_time.split(':')[0])
                minute = int(start_time.split(':')[1]) if ':' in start_time else 0
                
                # Clasificar por período del día con más precisión
                if hour < 10:
                    doc.metadata['time_period'] = 'temprano'
                    doc.metadata['time_period_en'] = 'early'
                elif hour < 12:
                    doc.metadata['time_period'] = 'mañana'
                    doc.metadata['time_period_en'] = 'morning'
                elif hour < 14:
                    doc.metadata['time_period'] = 'mediodía'
                    doc.metadata['time_period_en'] = 'midday'
                elif hour < 17:
                    doc.metadata['time_period'] = 'tarde'
                    doc.metadata['time_period_en'] = 'afternoon'
                else:
                    doc.metadata['time_period'] = 'noche'
                    doc.metadata['time_period_en'] = 'evening'
                
                # Agregar hora formateada
                doc.metadata['formatted_time'] = f"{hour:02d}:{minute:02d}"
            except:
                pass
    
    def _enrich_speaker_metadata(self, doc: Document):
        """Enriquece con variaciones de nombres de speakers"""
        speaker_names = doc.metadata.get('speaker_names', '')
        if speaker_names:
            # Crear variaciones de nombres (con/sin tildes, apellidos, etc.)
            names_variations = set()
            speakers = speaker_names.split(',')
            
            for speaker in speakers:
                speaker = speaker.strip()
                if speaker:
                    names_variations.add(speaker.lower())
                    # Quitar tildes
                    speaker_no_accents = self._remove_accents(speaker)
                    names_variations.add(speaker_no_accents.lower())
                    
                    # Agregar solo nombres (sin apellidos)
                    name_parts = speaker.split()
                    if len(name_parts) >= 2:
                        first_name = name_parts[0]
                        names_variations.add(first_name.lower())
                        names_variations.add(self._remove_accents(first_name).lower())
                        
                        # Agregar combinaciones de nombres
                        if len(name_parts) >= 3:
                            first_two = f"{name_parts[0]} {name_parts[1]}"
                            names_variations.add(first_two.lower())
                            names_variations.add(self._remove_accents(first_two).lower())
            
            doc.metadata['speaker_variations'] = list(names_variations)
    
    def _enrich_location_metadata(self, doc: Document):
        """Enriquece con información de ubicación normalizada"""
        room = doc.metadata.get('room', '')
        if room:
            # Normalizar nombres de salones
            room_lower = room.lower()
            if 'landívar' in room_lower or 'landivar' in room_lower:
                doc.metadata['room_normalized'] = 'landivar'
                doc.metadata['room_variations'] = ['landivar', 'landívar', 'salón landívar', 'salon landivar']
            elif 'obispo' in room_lower:
                doc.metadata['room_normalized'] = 'obispo'
                doc.metadata['room_variations'] = ['obispo', 'el obispo', 'salón el obispo', 'salon el obispo']
            elif 'pedro' in room_lower:
                doc.metadata['room_normalized'] = 'don-pedro'
                doc.metadata['room_variations'] = ['don pedro', 'don-pedro', 'salón don pedro', 'salon don pedro']
    
    def _remove_accents(self, text: str) -> str:
        """Remueve acentos de un texto"""
        accent_map = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ñ': 'n', 'Ñ': 'N'
        }
        for accented, unaccented in accent_map.items():
            text = text.replace(accented, unaccented)
        return text
    
    def _build_enhanced_indices(self):
        """Construye índices optimizados para la nueva estructura"""
        self.indices = {
            'by_speaker': {},
            'by_speaker_variations': {},
            'by_session': {},
            'by_time_period': {'temprano': [], 'mañana': [], 'mediodía': [], 'tarde': [], 'noche': []},
            'by_room': {},
            'by_topic_advanced': {},
            'by_company': {},
            'by_session_type': {},
            'by_hour': {}
        }
        
        for doc in self.documents:
            metadata = doc.metadata
            
            # Índice por speaker (mejorado)
            if 'speaker_names' in metadata:
                speakers = metadata['speaker_names'].split(',')
                for speaker in speakers:
                    speaker = speaker.strip()
                    if speaker:
                        if speaker not in self.indices['by_speaker']:
                            self.indices['by_speaker'][speaker] = []
                        self.indices['by_speaker'][speaker].append(doc)
            
            # Índice por variaciones de speaker
            if 'speaker_variations' in metadata:
                for variation in metadata['speaker_variations']:
                    if variation not in self.indices['by_speaker_variations']:
                        self.indices['by_speaker_variations'][variation] = []
                    self.indices['by_speaker_variations'][variation].append(doc)
            
            # Índice por sesión
            if 'session_name' in metadata:
                session = metadata['session_name']
                self.indices['by_session'][session] = doc
            
            # Índice por período del día (mejorado)
            time_period = metadata.get('time_period')
            if time_period in self.indices['by_time_period']:
                self.indices['by_time_period'][time_period].append(doc)
            
            # Índice por sala normalizada
            room_normalized = metadata.get('room_normalized')
            if room_normalized:
                if room_normalized not in self.indices['by_room']:
                    self.indices['by_room'][room_normalized] = []
                self.indices['by_room'][room_normalized].append(doc)
            
            # Índice por empresa
            company = metadata.get('company', '')
            if company:
                if company not in self.indices['by_company']:
                    self.indices['by_company'][company] = []
                self.indices['by_company'][company].append(doc)
            
            # Índice por tipo de sesión
            session_type = metadata.get('session_type', '')
            if session_type:
                if session_type not in self.indices['by_session_type']:
                    self.indices['by_session_type'][session_type] = []
                self.indices['by_session_type'][session_type].append(doc)
            
            # Índice por hora específica
            start_time = metadata.get('start_time', '')
            if start_time:
                hour = start_time.split(':')[0] if ':' in start_time else start_time
                if hour not in self.indices['by_hour']:
                    self.indices['by_hour'][hour] = []
                self.indices['by_hour'][hour].append(doc)
            
            # Índice de temas avanzado (usando el nuevo sistema de tags)
            content_lower = doc.page_content.lower()
            session_name_lower = metadata.get('session_name', '').lower()
            
            # Temas específicos basados en los nuevos tags
            advanced_topics = {
                'inteligencia-artificial': [
                    'inteligencia artificial', 'ia', 'ai', 'artificial intelligence',
                    'chatbot', 'llm', 'machine learning', 'ml', 'deep learning',
                    'langchain', 'kubeflow', 'granite', 'computer vision', 
                    'vision computacional', 'mlops', 'databricks'
                ],
                'seguridad': [
                    'seguridad', 'security', 'oauth', 'auth0', 'authentication',
                    'authorization', 'rbac', 'compliance', 'blindado', 'hardening',
                    'vulnerabilidades', 'api security', 'network security', 'ebpf', 'cilium'
                ],
                'kubernetes': [
                    'kubernetes', 'k8s', 'helm', 'keda', 'autoscaling', 'pods',
                    'containers', 'docker', 'namespaces', 'ingress', 'kubevirt'
                ],
                'devops': [
                    'devops', 'ci/cd', 'gitops', 'jenkins', 'argocd', 'pipeline',
                    'deployment', 'automation', 'github actions', 'canary', 'blue-green'
                ],
                'cloud-native': [
                    'cloud native', 'cncf', 'microservices', 'serverless', 'knative',
                    'istio', 'service mesh', 'observability', 'prometheus', 'grafana'
                ],
                'infraestructura': [
                    'infrastructure', 'terraform', 'ansible', 'kops', 'provisioning',
                    'iac', 'infrastructure as code', 'crossplane'
                ],
                'desarrollo': [
                    'development', 'vscode', 'backstage', 'ide', 'programming',
                    'testing', 'debugging'
                ]
            }
            
            for topic, keywords in advanced_topics.items():
                if any(keyword in content_lower or keyword in session_name_lower for keyword in keywords):
                    if topic not in self.indices['by_topic_advanced']:
                        self.indices['by_topic_advanced'][topic] = []
                    self.indices['by_topic_advanced'][topic].append(doc)


class OptimizedEventRetriever(BaseRetriever):
    """Retriever optimizado para la nueva estructura de base de datos"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    knowledge_base: EnhancedEventKnowledgeBase = Field(...)
    cross_encoder: Optional[CrossEncoder] = Field(default=None)
    
    def __init__(self, knowledge_base: EnhancedEventKnowledgeBase, **kwargs):
        super().__init__(knowledge_base=knowledge_base, **kwargs)
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except:
            logger.warning("CrossEncoder no disponible")
            self.cross_encoder = None
    
    def _extract_entities_enhanced(self, query: str) -> Dict[str, Any]:
        """Extrae entidades con mayor precisión usando el nuevo sistema"""
        entities = {
            'names': [],
            'topics': [],
            'time_queries': [],
            'location_queries': [],
            'companies': [],
            'query_type': 'general',
            'original_query': query,
            'specific_time': None,
            'room_names': []
        }
        
        query_lower = query.lower()
        query_no_accents = self._remove_accents(query_lower)
        
        # PRIMERO: Extraer temas específicos antes de detectar el tipo
        topic_patterns = [
            r'(?:qué\s+charlas?|que\s+charlas?)\s+(?:hay\s+)?(?:sobre|de)\s+([a-záéíóúñ\s]+?)(?:\s+hay)?[?]?',
            r'(?:charlas?|sesion(?:es)?)\s+(?:sobre|de)\s+([a-záéíóúñ\s]+?)(?:\s+hay)?[?]?',
            r'(?:tema|topic)\s+(?:de|sobre)\s+([a-záéíóúñ\s]+?)(?:\s+hay)?[?]?',
            r'(?:hay|existe|tienen)\s+(?:charlas?|sesion(?:es)?)\s+(?:sobre|de)\s+([a-záéíóúñ\s]+?)[?]?'
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                clean_topic = match.strip().rstrip('?').strip()
                # Limpiar palabras extra que no son parte del tema
                clean_topic = re.sub(r'\s+(hay|existe|tienen)
        
        # SEGUNDO: Detectar tipos de consulta más específicos y precisos
        if any(word in query_lower for word in ['cuántas', 'cuantas', 'total', 'cantidad', 'número', 'numero']):
            entities['query_type'] = 'count'
        elif any(word in query_lower for word in ['qué hora', 'que hora', 'hora', 'horario', 'cuándo', 'cuando', 'empieza', 'termina']):
            entities['query_type'] = 'time'
        elif any(word in query_lower for word in ['dónde', 'donde', 'salón', 'salon', 'lugar', 'ubicación', 'sala']):
            entities['query_type'] = 'location'
        elif any(word in query_lower for word in ['quién', 'quien', 'de quién', 'de quien', 'ponente', 'speaker']):
            entities['query_type'] = 'speaker'
        elif any(word in query_lower for word in ['tema', 'trata sobre', 'sobre qué', 'sobre que', 'tema principal', 'contenido']):
            entities['query_type'] = 'topic'
        # PRIORIDAD ALTA: Si ya encontramos temas, es topic_search
        elif entities['topics'] or any(word in query_lower for word in ['qué charlas', 'que charlas', 'charlas sobre', 'charlas de', 'sesiones sobre', 'sesiones de', 'hay charlas']):
            entities['query_type'] = 'topic_search'
        elif any(word in query_lower for word in ['mañana', 'tarde', 'morning', 'afternoon', 'mediodía', 'mediodia']):
            entities['query_type'] = 'time_period'
        elif any(word in query_lower for word in ['duración', 'duracion', 'tiempo', 'cuánto dura', 'cuanto dura', 'termina']):
            entities['query_type'] = 'duration'
        elif any(word in query_lower for word in ['cual es la charla', 'qué charla', 'que charla', 'charla de']):
            entities['query_type'] = 'session_by_speaker'
        
        # TERCERO: Extraer nombres SOLO si no es una consulta de temas
        if entities['query_type'] not in ['topic_search', 'topic']:
            words = query.split()
            i = 0
            while i < len(words):
                word = words[i]
                # Filtrar palabras que NO son nombres pero están en mayúscula
                if (len(word) > 2 and word[0].isupper() and 
                    word not in ['El', 'La', 'Los', 'Las', 'De', 'Del', 'En', 'Por', 'Para', 'Cual', 'Que', 'IA?', 'AI?'] and
                    not word.endswith('?')):
                    # Buscar nombres compuestos
                    name_parts = [word]
                    j = i + 1
                    while j < len(words) and len(words[j]) > 1 and (words[j][0].isupper() or words[j].lower() in ['de', 'del', 'la']):
                        if not words[j].endswith('?'):
                            name_parts.append(words[j])
                        j += 1
                    
                    full_name = ' '.join(name_parts)
                    entities['names'].append(full_name)
                    # Agregar variación sin acentos
                    entities['names'].append(self._remove_accents(full_name))
                    i = j
                else:
                    i += 1
        
        # CUARTO: Si no se detectaron temas con regex, buscar temas conocidos en la consulta
        if not entities['topics'] and entities['query_type'] == 'topic_search':
            known_topics = {
                'ia': ['ia', 'ai', 'inteligencia artificial', 'artificial intelligence'],
                'seguridad': ['seguridad', 'security', 'oauth', 'auth'],
                'kubernetes': ['kubernetes', 'k8s', 'helm', 'keda'],
                'devops': ['devops', 'ci/cd', 'gitops', 'deployment'],
                'machine learning': ['machine learning', 'ml', 'mlops', 'databricks'],
                'desarrollo': ['desarrollo', 'development', 'vscode', 'backstage']
            }
            
            for topic, keywords in known_topics.items():
                if any(keyword in query_lower for keyword in keywords):
                    entities['topics'].append(topic)
                    break
        
        # Extraer nombres de salas específicos
        room_patterns = [
            r'(?:salón|salon)\s+(landívar|landivar|el\s+obispo|don\s+pedro)',
            r'(landívar|landivar|obispo|pedro)',
        ]
        for pattern in room_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match).strip()
                entities['room_names'].append(match)
        
        # Extraer hora específica
        time_pattern = r'(\d{1,2}):?(\d{2})?'
        time_matches = re.findall(time_pattern, query)
        if time_matches:
            hour, minute = time_matches[0]
            entities['specific_time'] = f"{hour}:{minute if minute else '00'}"
        
        # Detectar períodos de tiempo específicos
        if any(word in query_lower for word in ['mañana', 'morning']):
            entities['time_queries'].append('mañana')
        if any(word in query_lower for word in ['tarde', 'afternoon']):
            entities['time_queries'].append('tarde')
        if any(word in query_lower for word in ['mediodía', 'mediodia', 'midday']):
            entities['time_queries'].append('mediodía')
        
        # Detectar empresas específicas
        companies = ['red hat', 'redhat', 'telus', 'gbm', 'usac', 'walmart', 'replicated', 'kong', 'bantrab']
        for company in companies:
            if company in query_lower:
                entities['companies'].append(company)
        
        return entities
    
    def _remove_accents(self, text: str) -> str:
        """Remueve acentos de un texto"""
        accent_map = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ñ': 'n', 'Ñ': 'N'
        }
        for accented, unaccented in accent_map.items():
            text = text.replace(accented, unaccented)
        return text
    
    def _search_by_speaker_enhanced(self, names: List[str]) -> List[Document]:
        """Búsqueda mejorada por speaker usando variaciones"""
        results = []
        seen_sessions = set()
        
        for name in names:
            name_lower = name.lower()
            name_no_accents = self._remove_accents(name_lower)
            
            # Búsqueda en variaciones de speakers
            for variation in [name_lower, name_no_accents]:
                if variation in self.knowledge_base.indices['by_speaker_variations']:
                    for doc in self.knowledge_base.indices['by_speaker_variations'][variation]:
                        session = doc.metadata.get('session_name', '')
                        if session and session not in seen_sessions:
                            seen_sessions.add(session)
                            results.append(doc)
            
            # Búsqueda parcial mejorada
            for speaker_var, docs in self.knowledge_base.indices['by_speaker_variations'].items():
                if (name_lower in speaker_var or 
                    name_no_accents in speaker_var or
                    any(part in speaker_var for part in name_lower.split() if len(part) > 2)):
                    for doc in docs:
                        session = doc.metadata.get('session_name', '')
                        if session and session not in seen_sessions:
                            seen_sessions.add(session)
                            results.append(doc)
        
        return results
    
    def _search_by_topic_enhanced(self, topics: List[str]) -> List[Document]:
        """Búsqueda mejorada por tema usando el nuevo sistema de tags"""
        all_results = []
        seen_ids = set()
        
        for topic in topics:
            topic_lower = topic.lower()
            topic_no_accents = self._remove_accents(topic_lower)
            
            # Buscar en índice de temas avanzado
            for indexed_topic, docs in self.knowledge_base.indices['by_topic_advanced'].items():
                if (topic_lower in indexed_topic or 
                    topic_no_accents in indexed_topic or
                    indexed_topic in topic_lower or
                    indexed_topic in topic_no_accents):
                    for doc in docs:
                        doc_id = id(doc)
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            all_results.append(doc)
            
            # Búsqueda semántica adicional siempre (no solo si no hay resultados)
            semantic_query = f"charlas {topic} sesiones {topic} {topic_no_accents}"
            docs = self.knowledge_base.vector_store.similarity_search(semantic_query, k=30)
            for doc in docs:
                content_lower = doc.page_content.lower()
                session_name = doc.metadata.get('session_name', '').lower()
                
                if (topic_lower in content_lower or topic_lower in session_name or
                    topic_no_accents in content_lower or topic_no_accents in session_name):
                    doc_id = id(doc)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_results.append(doc)
        
        return all_results
    
    def _search_by_room_enhanced(self, room_names: List[str]) -> List[Document]:
        """Búsqueda por sala específica"""
        results = []
        for room_name in room_names:
            room_lower = room_name.lower()
            room_no_accents = self._remove_accents(room_lower)
            
            # Mapear a nombres normalizados
            if 'landívar' in room_lower or 'landivar' in room_lower:
                room_key = 'landivar'
            elif 'obispo' in room_lower:
                room_key = 'obispo'
            elif 'pedro' in room_lower:
                room_key = 'don-pedro'
            else:
                room_key = room_lower
            
            if room_key in self.knowledge_base.indices['by_room']:
                results.extend(self.knowledge_base.indices['by_room'][room_key])
        
        return results
    
    def _search_by_time_period_enhanced(self, periods: List[str]) -> List[Document]:
        """Búsqueda mejorada por período del día"""
        results = []
        for period in periods:
            if period in self.knowledge_base.indices['by_time_period']:
                results.extend(self.knowledge_base.indices['by_time_period'][period])
        return results
    
    def _search_by_specific_time(self, time_str: str) -> List[Document]:
        """Búsqueda por hora específica"""
        hour = time_str.split(':')[0]
        if hour in self.knowledge_base.indices['by_hour']:
            return self.knowledge_base.indices['by_hour'][hour]
        return []
    
    def _handle_count_query_enhanced(self, query: str, entities: Dict) -> List[Document]:
        """Maneja consultas de conteo con mayor precisión"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['charla', 'sesion', 'session', 'talk']):
            # Excluir breaks y actividades del conteo
            valid_sessions = [session for session in self.knowledge_base.indices['by_session'].keys() 
                            if not any(word in session.lower() for word in ['break', 'almuerzo', 'check-in', 'cierre', 'networking'])]
            count = len(valid_sessions)
            return [Document(
                page_content=f"{count}",
                metadata={"source": "count", "type": "sessions"}
            )]
        elif any(word in query_lower for word in ['ponente', 'speaker', 'presentador']):
            # Excluir "Staff" del conteo
            speakers = [speaker for speaker in self.knowledge_base.indices['by_speaker'].keys() 
                       if speaker.lower() != 'staff']
            count = len(speakers)
            return [Document(
                page_content=f"{count}",
                metadata={"source": "count", "type": "speakers"}
            )]
        
        return []
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Obtiene documentos relevantes con la lógica optimizada"""
        
        entities = self._extract_entities_enhanced(query)
        logger.info(f"Entidades extraídas: {entities}")
        
        # Manejar consultas de conteo
        if entities['query_type'] == 'count':
            return self._handle_count_query_enhanced(query, entities)
        
        # Búsqueda por hora específica
        elif entities['specific_time']:
            return self._search_by_specific_time(entities['specific_time'])
        
        # Búsqueda por sala específica
        elif entities['query_type'] == 'location' and entities['room_names']:
            return self._search_by_room_enhanced(entities['room_names'])
        
        # Búsqueda por período del día
        elif entities['query_type'] == 'time_period' and entities['time_queries']:
            return self._search_by_time_period_enhanced(entities['time_queries'])
        
        # Preguntas sobre tema/contenido de una charla específica por speaker
        elif entities['query_type'] in ['topic', 'session_by_speaker'] and entities['names']:
            docs = self._search_by_speaker_enhanced(entities['names'])
            return docs[:1] if docs else []
        
        # Búsqueda por speaker para horarios, ubicación, etc.
        elif entities['names'] and entities['query_type'] in ['time', 'location', 'speaker', 'duration']:
            return self._search_by_speaker_enhanced(entities['names'])
        
        # Búsqueda por tema (charlas que tratan sobre X)
        elif entities['query_type'] == 'topic_search' and entities['topics']:
            return self._search_by_topic_enhanced(entities['topics'])
        
        # Búsqueda general con nombres
        elif entities['names']:
            return self._search_by_speaker_enhanced(entities['names'])
        
        # Búsqueda por empresa
        elif entities['companies']:
            results = []
            for company in entities['companies']:
                if company in self.knowledge_base.indices['by_company']:
                    results.extend(self.knowledge_base.indices['by_company'][company])
            if results:
                return results
        
        # Búsqueda semántica general mejorada
        docs = self.knowledge_base.vector_store.similarity_search(query, k=15)
        
        # Re-ranking si está disponible
        if self.cross_encoder and len(docs) > 1:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            docs = [doc for _, doc in ranked_docs]
        
        return docs[:8]


def format_docs_for_llm_optimized(docs: List[Document]) -> str:
    """Formatea documentos optimizado para la nueva estructura"""
    if not docs:
        return "No hay información disponible."
    
    # Para resultados de conteo
    if len(docs) == 1 and docs[0].metadata.get("source") == "count":
        return docs[0].page_content
    
    # Para múltiples charlas relacionadas con un tema (como seguridad)
    if len(docs) > 1:
        formatted_talks = []
        seen_sessions = set()
        
        for doc in docs:
            session_name = doc.metadata.get('session_name', '').strip()
            speaker_names = doc.metadata.get('speaker_names', '').strip()
            start_time = doc.metadata.get('start_time', '').strip()
            room = doc.metadata.get('room', '').strip()
            
            if session_name and session_name not in seen_sessions:
                seen_sessions.add(session_name)
                
                # Formato simple para listas
                talk_info = []
                if session_name:
                    talk_info.append(f'"{session_name}"')
                if speaker_names:
                    talk_info.append(f"por {speaker_names}")
                if start_time:
                    talk_info.append(f"a las {start_time}")
                if room:
                    talk_info.append(f"en {room}")
                
                if talk_info:
                    formatted_talks.append(" ".join(talk_info))
        
        if len(formatted_talks) > 1:
            return "\n".join([f"• {talk}" for talk in formatted_talks])
        elif len(formatted_talks) == 1:
            return formatted_talks[0]
    
    # Para una sola charla
    if len(docs) == 1:
        doc = docs[0]
        session_name = doc.metadata.get('session_name', '').strip()
        speaker_names = doc.metadata.get('speaker_names', '').strip()
        start_time = doc.metadata.get('start_time', '').strip()
        room = doc.metadata.get('room', '').strip()
        
        # Formato simple para respuesta única
        talk_info = []
        if session_name:
            talk_info.append(f'"{session_name}"')
        if speaker_names:
            talk_info.append(f"por {speaker_names}")
        if start_time:
            talk_info.append(f"a las {start_time}")
        if room:
            talk_info.append(f"en {room}")
        
        if talk_info:
            return " ".join(talk_info)
    
    return "No hay información específica disponible."


# Variables globales
ENHANCED_KNOWLEDGE_BASE: Optional[EnhancedEventKnowledgeBase] = None


def initialize_knowledge_base():
    """Inicializa la base de conocimiento optimizada para la nueva estructura"""
    global ENHANCED_KNOWLEDGE_BASE
    if ENHANCED_KNOWLEDGE_BASE is None:
        logger.info("Inicializando base de conocimiento optimizada para KCD Guatemala 2025...")
        try:
            db_type = os.getenv("DB_TYPE", "PGVECTOR")
            db_factory = DBFactory()
            base_retriever = db_factory.get_retriever(db_type)
            embeddings = base_retriever.vectorstore.embeddings
            
            # Recuperar documentos con mayor cantidad para aprovechar la nueva estructura
            all_docs = base_retriever.vectorstore.similarity_search(query=" ", k=3000)
            logger.info(f"Recuperados {len(all_docs)} documentos de la nueva base de datos")
            
            ENHANCED_KNOWLEDGE_BASE = EnhancedEventKnowledgeBase(all_docs, embeddings)
            
            # Log de estadísticas de la base de conocimiento
            logger.info(f"Índices construidos:")
            logger.info(f"- Speakers: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_speaker'])}")
            logger.info(f"- Variaciones de speakers: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_speaker_variations'])}")
            logger.info(f"- Sesiones: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_session'])}")
            logger.info(f"- Temas avanzados: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_topic_advanced'])}")
            logger.info(f"- Salas: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_room'])}")
            
        except Exception as e:
            logger.error(f"Error inicializando la base de conocimiento optimizada: {e}")
            raise
    return ENHANCED_KNOWLEDGE_BASE


def create_qa_chain(knowledge_base: EnhancedEventKnowledgeBase, llm: Any) -> Runnable:
    """Crea la cadena QA optimizada para la nueva estructura"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    
    chain = (
        RunnableParallel(
            context=(lambda x: x['input']) | retriever | format_docs_for_llm_optimized,
            input=(lambda x: x['input']),
            event_name=lambda x: EVENT_NAME
        )
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return chain


# Funciones auxiliares para análisis y debugging
def analyze_query_performance(query: str, knowledge_base: EnhancedEventKnowledgeBase) -> Dict[str, Any]:
    """Analiza el rendimiento de una consulta específica (para debugging)"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    entities = retriever._extract_entities_enhanced(query)
    
    # Simular la búsqueda
    if entities['query_type'] == 'count':
        docs = retriever._handle_count_query_enhanced(query, entities)
    elif entities['names']:
        docs = retriever._search_by_speaker_enhanced(entities['names'])
    elif entities['topics']:
        docs = retriever._search_by_topic_enhanced(entities['topics'])
    elif entities['time_queries']:
        docs = retriever._search_by_time_period_enhanced(entities['time_queries'])
    elif entities['room_names']:
        docs = retriever._search_by_room_enhanced(entities['room_names'])
    else:
        docs = knowledge_base.vector_store.similarity_search(query, k=5)
    
    return {
        'query': query,
        'entities_extracted': entities,
        'documents_found': len(docs),
        'documents_preview': [
            {
                'session_name': doc.metadata.get('session_name', 'N/A'),
                'speaker_names': doc.metadata.get('speaker_names', 'N/A'),
                'start_time': doc.metadata.get('start_time', 'N/A')
            }
            for doc in docs[:3]
        ]
    }


def get_knowledge_base_stats(knowledge_base: EnhancedEventKnowledgeBase) -> Dict[str, Any]:
    """Obtiene estadísticas de la base de conocimiento (para debugging)"""
    return {
        'total_documents': len(knowledge_base.documents),
        'unique_speakers': len(knowledge_base.indices['by_speaker']),
        'speaker_variations': len(knowledge_base.indices['by_speaker_variations']),
        'unique_sessions': len(knowledge_base.indices['by_session']),
        'topics_indexed': len(knowledge_base.indices['by_topic_advanced']),
        'rooms_indexed': len(knowledge_base.indices['by_room']),
        'companies_indexed': len(knowledge_base.indices['by_company']),
        'time_periods': {
            period: len(docs) 
            for period, docs in knowledge_base.indices['by_time_period'].items()
        },
        'sample_speakers': list(knowledge_base.indices['by_speaker'].keys())[:10],
        'sample_topics': list(knowledge_base.indices['by_topic_advanced'].keys()),
        'sample_rooms': list(knowledge_base.indices['by_room'].keys())
    }


# Funciones de utilidad para testing específico
def test_speaker_search(speaker_name: str, knowledge_base: EnhancedEventKnowledgeBase) -> List[str]:
    """Prueba la búsqueda de un speaker específico"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    docs = retriever._search_by_speaker_enhanced([speaker_name])
    return [doc.metadata.get('session_name', 'N/A') for doc in docs]


def test_topic_search(topic: str, knowledge_base: EnhancedEventKnowledgeBase) -> List[str]:
    """Prueba la búsqueda de un tema específico"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    docs = retriever._search_by_topic_enhanced([topic])
    return [doc.metadata.get('session_name', 'N/A') for doc in docs]


def test_time_search(time_period: str, knowledge_base: EnhancedEventKnowledgeBase) -> List[str]:
    """Prueba la búsqueda por período de tiempo"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    docs = retriever._search_by_time_period_enhanced([time_period])
    return [
        f"{doc.metadata.get('session_name', 'N/A')} - {doc.metadata.get('start_time', 'N/A')}"
        for doc in docs
    ]


# Configuración de logging específica para debugging
def enable_debug_logging():
    """Habilita logging detallado para debugging"""
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logger.info("Debug logging habilitado para query_helper")


def disable_debug_logging():
    """Deshabilita logging detallado"""
    logging.getLogger(__name__).setLevel(logging.WARNING)
    logger.info("Debug logging deshabilitado para query_helper"), '', clean_topic).strip()
                if clean_topic and len(clean_topic) > 1:
                    entities['topics'].append(clean_topic)
        
        # SEGUNDO: Detectar tipos de consulta más específicos y precisos
        if any(word in query_lower for word in ['cuántas', 'cuantas', 'total', 'cantidad', 'número', 'numero']):
            entities['query_type'] = 'count'
        elif any(word in query_lower for word in ['qué hora', 'que hora', 'hora', 'horario', 'cuándo', 'cuando', 'empieza', 'termina']):
            entities['query_type'] = 'time'
        elif any(word in query_lower for word in ['dónde', 'donde', 'salón', 'salon', 'lugar', 'ubicación', 'sala']):
            entities['query_type'] = 'location'
        elif any(word in query_lower for word in ['quién', 'quien', 'de quién', 'de quien', 'ponente', 'speaker']):
            entities['query_type'] = 'speaker'
        elif any(word in query_lower for word in ['tema', 'trata sobre', 'sobre qué', 'sobre que', 'tema principal', 'contenido']):
            entities['query_type'] = 'topic'
        # PRIORIDAD ALTA: Si ya encontramos temas, es topic_search
        elif entities['topics'] or any(word in query_lower for word in ['qué charlas', 'que charlas', 'charlas sobre', 'charlas de', 'sesiones sobre', 'sesiones de', 'hay charlas']):
            entities['query_type'] = 'topic_search'
        elif any(word in query_lower for word in ['mañana', 'tarde', 'morning', 'afternoon', 'mediodía', 'mediodia']):
            entities['query_type'] = 'time_period'
        elif any(word in query_lower for word in ['duración', 'duracion', 'tiempo', 'cuánto dura', 'cuanto dura', 'termina']):
            entities['query_type'] = 'duration'
        elif any(word in query_lower for word in ['cual es la charla', 'qué charla', 'que charla', 'charla de']):
            entities['query_type'] = 'session_by_speaker'
        
        # TERCERO: Extraer nombres SOLO si no es una consulta de temas
        if entities['query_type'] not in ['topic_search', 'topic']:
            words = query.split()
            i = 0
            while i < len(words):
                word = words[i]
                # Filtrar palabras que NO son nombres pero están en mayúscula
                if (len(word) > 2 and word[0].isupper() and 
                    word not in ['El', 'La', 'Los', 'Las', 'De', 'Del', 'En', 'Por', 'Para', 'Cual', 'Que', 'IA?', 'AI?'] and
                    not word.endswith('?')):
                    # Buscar nombres compuestos
                    name_parts = [word]
                    j = i + 1
                    while j < len(words) and len(words[j]) > 1 and (words[j][0].isupper() or words[j].lower() in ['de', 'del', 'la']):
                        if not words[j].endswith('?'):
                            name_parts.append(words[j])
                        j += 1
                    
                    full_name = ' '.join(name_parts)
                    entities['names'].append(full_name)
                    # Agregar variación sin acentos
                    entities['names'].append(self._remove_accents(full_name))
                    i = j
                else:
                    i += 1
        
        # CUARTO: Si no se detectaron temas con regex, buscar temas conocidos en la consulta
        if not entities['topics'] and entities['query_type'] == 'topic_search':
            known_topics = {
                'ia': ['ia', 'ai', 'inteligencia artificial', 'artificial intelligence'],
                'seguridad': ['seguridad', 'security', 'oauth', 'auth'],
                'kubernetes': ['kubernetes', 'k8s', 'helm', 'keda'],
                'devops': ['devops', 'ci/cd', 'gitops', 'deployment'],
                'machine learning': ['machine learning', 'ml', 'mlops', 'databricks'],
                'desarrollo': ['desarrollo', 'development', 'vscode', 'backstage']
            }
            
            for topic, keywords in known_topics.items():
                if any(keyword in query_lower for keyword in keywords):
                    entities['topics'].append(topic)
                    break
        
        # Extraer nombres de salas específicos
        room_patterns = [
            r'(?:salón|salon)\s+(landívar|landivar|el\s+obispo|don\s+pedro)',
            r'(landívar|landivar|obispo|pedro)',
        ]
        for pattern in room_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match).strip()
                entities['room_names'].append(match)
        
        # Extraer hora específica
        time_pattern = r'(\d{1,2}):?(\d{2})?'
        time_matches = re.findall(time_pattern, query)
        if time_matches:
            hour, minute = time_matches[0]
            entities['specific_time'] = f"{hour}:{minute if minute else '00'}"
        
        # Detectar períodos de tiempo específicos
        if any(word in query_lower for word in ['mañana', 'morning']):
            entities['time_queries'].append('mañana')
        if any(word in query_lower for word in ['tarde', 'afternoon']):
            entities['time_queries'].append('tarde')
        if any(word in query_lower for word in ['mediodía', 'mediodia', 'midday']):
            entities['time_queries'].append('mediodía')
        
        # Detectar empresas específicas
        companies = ['red hat', 'redhat', 'telus', 'gbm', 'usac', 'walmart', 'replicated', 'kong', 'bantrab']
        for company in companies:
            if company in query_lower:
                entities['companies'].append(company)
        
        return entities
    
    def _remove_accents(self, text: str) -> str:
        """Remueve acentos de un texto"""
        accent_map = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ñ': 'n', 'Ñ': 'N'
        }
        for accented, unaccented in accent_map.items():
            text = text.replace(accented, unaccented)
        return text
    
    def _search_by_speaker_enhanced(self, names: List[str]) -> List[Document]:
        """Búsqueda mejorada por speaker usando variaciones"""
        results = []
        seen_sessions = set()
        
        for name in names:
            name_lower = name.lower()
            name_no_accents = self._remove_accents(name_lower)
            
            # Búsqueda en variaciones de speakers
            for variation in [name_lower, name_no_accents]:
                if variation in self.knowledge_base.indices['by_speaker_variations']:
                    for doc in self.knowledge_base.indices['by_speaker_variations'][variation]:
                        session = doc.metadata.get('session_name', '')
                        if session and session not in seen_sessions:
                            seen_sessions.add(session)
                            results.append(doc)
            
            # Búsqueda parcial mejorada
            for speaker_var, docs in self.knowledge_base.indices['by_speaker_variations'].items():
                if (name_lower in speaker_var or 
                    name_no_accents in speaker_var or
                    any(part in speaker_var for part in name_lower.split() if len(part) > 2)):
                    for doc in docs:
                        session = doc.metadata.get('session_name', '')
                        if session and session not in seen_sessions:
                            seen_sessions.add(session)
                            results.append(doc)
        
        return results
    
    def _search_by_topic_enhanced(self, topics: List[str]) -> List[Document]:
        """Búsqueda mejorada por tema usando el nuevo sistema de tags"""
        all_results = []
        seen_ids = set()
        
        for topic in topics:
            topic_lower = topic.lower()
            topic_no_accents = self._remove_accents(topic_lower)
            
            # Buscar en índice de temas avanzado
            for indexed_topic, docs in self.knowledge_base.indices['by_topic_advanced'].items():
                if (topic_lower in indexed_topic or 
                    topic_no_accents in indexed_topic or
                    indexed_topic in topic_lower or
                    indexed_topic in topic_no_accents):
                    for doc in docs:
                        doc_id = id(doc)
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            all_results.append(doc)
            
            # Búsqueda semántica adicional siempre (no solo si no hay resultados)
            semantic_query = f"charlas {topic} sesiones {topic} {topic_no_accents}"
            docs = self.knowledge_base.vector_store.similarity_search(semantic_query, k=30)
            for doc in docs:
                content_lower = doc.page_content.lower()
                session_name = doc.metadata.get('session_name', '').lower()
                
                if (topic_lower in content_lower or topic_lower in session_name or
                    topic_no_accents in content_lower or topic_no_accents in session_name):
                    doc_id = id(doc)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_results.append(doc)
        
        return all_results
    
    def _search_by_room_enhanced(self, room_names: List[str]) -> List[Document]:
        """Búsqueda por sala específica"""
        results = []
        for room_name in room_names:
            room_lower = room_name.lower()
            room_no_accents = self._remove_accents(room_lower)
            
            # Mapear a nombres normalizados
            if 'landívar' in room_lower or 'landivar' in room_lower:
                room_key = 'landivar'
            elif 'obispo' in room_lower:
                room_key = 'obispo'
            elif 'pedro' in room_lower:
                room_key = 'don-pedro'
            else:
                room_key = room_lower
            
            if room_key in self.knowledge_base.indices['by_room']:
                results.extend(self.knowledge_base.indices['by_room'][room_key])
        
        return results
    
    def _search_by_time_period_enhanced(self, periods: List[str]) -> List[Document]:
        """Búsqueda mejorada por período del día"""
        results = []
        for period in periods:
            if period in self.knowledge_base.indices['by_time_period']:
                results.extend(self.knowledge_base.indices['by_time_period'][period])
        return results
    
    def _search_by_specific_time(self, time_str: str) -> List[Document]:
        """Búsqueda por hora específica"""
        hour = time_str.split(':')[0]
        if hour in self.knowledge_base.indices['by_hour']:
            return self.knowledge_base.indices['by_hour'][hour]
        return []
    
    def _handle_count_query_enhanced(self, query: str, entities: Dict) -> List[Document]:
        """Maneja consultas de conteo con mayor precisión"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['charla', 'sesion', 'session', 'talk']):
            # Excluir breaks y actividades del conteo
            valid_sessions = [session for session in self.knowledge_base.indices['by_session'].keys() 
                            if not any(word in session.lower() for word in ['break', 'almuerzo', 'check-in', 'cierre', 'networking'])]
            count = len(valid_sessions)
            return [Document(
                page_content=f"{count}",
                metadata={"source": "count", "type": "sessions"}
            )]
        elif any(word in query_lower for word in ['ponente', 'speaker', 'presentador']):
            # Excluir "Staff" del conteo
            speakers = [speaker for speaker in self.knowledge_base.indices['by_speaker'].keys() 
                       if speaker.lower() != 'staff']
            count = len(speakers)
            return [Document(
                page_content=f"{count}",
                metadata={"source": "count", "type": "speakers"}
            )]
        
        return []
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Obtiene documentos relevantes con la lógica optimizada"""
        
        entities = self._extract_entities_enhanced(query)
        logger.info(f"Entidades extraídas: {entities}")
        
        # Manejar consultas de conteo
        if entities['query_type'] == 'count':
            return self._handle_count_query_enhanced(query, entities)
        
        # Búsqueda por hora específica
        elif entities['specific_time']:
            return self._search_by_specific_time(entities['specific_time'])
        
        # Búsqueda por sala específica
        elif entities['query_type'] == 'location' and entities['room_names']:
            return self._search_by_room_enhanced(entities['room_names'])
        
        # Búsqueda por período del día
        elif entities['query_type'] == 'time_period' and entities['time_queries']:
            return self._search_by_time_period_enhanced(entities['time_queries'])
        
        # Preguntas sobre tema/contenido de una charla específica por speaker
        elif entities['query_type'] in ['topic', 'session_by_speaker'] and entities['names']:
            docs = self._search_by_speaker_enhanced(entities['names'])
            return docs[:1] if docs else []
        
        # Búsqueda por speaker para horarios, ubicación, etc.
        elif entities['names'] and entities['query_type'] in ['time', 'location', 'speaker', 'duration']:
            return self._search_by_speaker_enhanced(entities['names'])
        
        # Búsqueda por tema (charlas que tratan sobre X)
        elif entities['query_type'] == 'topic_search' and entities['topics']:
            return self._search_by_topic_enhanced(entities['topics'])
        
        # Búsqueda general con nombres
        elif entities['names']:
            return self._search_by_speaker_enhanced(entities['names'])
        
        # Búsqueda por empresa
        elif entities['companies']:
            results = []
            for company in entities['companies']:
                if company in self.knowledge_base.indices['by_company']:
                    results.extend(self.knowledge_base.indices['by_company'][company])
            if results:
                return results
        
        # Búsqueda semántica general mejorada
        docs = self.knowledge_base.vector_store.similarity_search(query, k=15)
        
        # Re-ranking si está disponible
        if self.cross_encoder and len(docs) > 1:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            docs = [doc for _, doc in ranked_docs]
        
        return docs[:8]


def format_docs_for_llm_optimized(docs: List[Document]) -> str:
    """Formatea documentos optimizado para la nueva estructura"""
    if not docs:
        return "No hay información disponible."
    
    # Para resultados de conteo
    if len(docs) == 1 and docs[0].metadata.get("source") == "count":
        return docs[0].page_content
    
    # Para múltiples charlas relacionadas con un tema (como seguridad)
    if len(docs) > 1:
        formatted_talks = []
        seen_sessions = set()
        
        for doc in docs:
            session_name = doc.metadata.get('session_name', '').strip()
            speaker_names = doc.metadata.get('speaker_names', '').strip()
            start_time = doc.metadata.get('start_time', '').strip()
            room = doc.metadata.get('room', '').strip()
            
            if session_name and session_name not in seen_sessions:
                seen_sessions.add(session_name)
                
                # Formato simple para listas
                talk_info = []
                if session_name:
                    talk_info.append(f'"{session_name}"')
                if speaker_names:
                    talk_info.append(f"por {speaker_names}")
                if start_time:
                    talk_info.append(f"a las {start_time}")
                if room:
                    talk_info.append(f"en {room}")
                
                if talk_info:
                    formatted_talks.append(" ".join(talk_info))
        
        if len(formatted_talks) > 1:
            return "\n".join([f"• {talk}" for talk in formatted_talks])
        elif len(formatted_talks) == 1:
            return formatted_talks[0]
    
    # Para una sola charla
    if len(docs) == 1:
        doc = docs[0]
        session_name = doc.metadata.get('session_name', '').strip()
        speaker_names = doc.metadata.get('speaker_names', '').strip()
        start_time = doc.metadata.get('start_time', '').strip()
        room = doc.metadata.get('room', '').strip()
        
        # Formato simple para respuesta única
        talk_info = []
        if session_name:
            talk_info.append(f'"{session_name}"')
        if speaker_names:
            talk_info.append(f"por {speaker_names}")
        if start_time:
            talk_info.append(f"a las {start_time}")
        if room:
            talk_info.append(f"en {room}")
        
        if talk_info:
            return " ".join(talk_info)
    
    return "No hay información específica disponible."


# Variables globales
ENHANCED_KNOWLEDGE_BASE: Optional[EnhancedEventKnowledgeBase] = None


def initialize_knowledge_base():
    """Inicializa la base de conocimiento optimizada para la nueva estructura"""
    global ENHANCED_KNOWLEDGE_BASE
    if ENHANCED_KNOWLEDGE_BASE is None:
        logger.info("Inicializando base de conocimiento optimizada para KCD Guatemala 2025...")
        try:
            db_type = os.getenv("DB_TYPE", "PGVECTOR")
            db_factory = DBFactory()
            base_retriever = db_factory.get_retriever(db_type)
            embeddings = base_retriever.vectorstore.embeddings
            
            # Recuperar documentos con mayor cantidad para aprovechar la nueva estructura
            all_docs = base_retriever.vectorstore.similarity_search(query=" ", k=3000)
            logger.info(f"Recuperados {len(all_docs)} documentos de la nueva base de datos")
            
            ENHANCED_KNOWLEDGE_BASE = EnhancedEventKnowledgeBase(all_docs, embeddings)
            
            # Log de estadísticas de la base de conocimiento
            logger.info(f"Índices construidos:")
            logger.info(f"- Speakers: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_speaker'])}")
            logger.info(f"- Variaciones de speakers: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_speaker_variations'])}")
            logger.info(f"- Sesiones: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_session'])}")
            logger.info(f"- Temas avanzados: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_topic_advanced'])}")
            logger.info(f"- Salas: {len(ENHANCED_KNOWLEDGE_BASE.indices['by_room'])}")
            
        except Exception as e:
            logger.error(f"Error inicializando la base de conocimiento optimizada: {e}")
            raise
    return ENHANCED_KNOWLEDGE_BASE


def create_qa_chain(knowledge_base: EnhancedEventKnowledgeBase, llm: Any) -> Runnable:
    """Crea la cadena QA optimizada para la nueva estructura"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    
    chain = (
        RunnableParallel(
            context=(lambda x: x['input']) | retriever | format_docs_for_llm_optimized,
            input=(lambda x: x['input']),
            event_name=lambda x: EVENT_NAME
        )
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return chain


# Funciones auxiliares para análisis y debugging
def analyze_query_performance(query: str, knowledge_base: EnhancedEventKnowledgeBase) -> Dict[str, Any]:
    """Analiza el rendimiento de una consulta específica (para debugging)"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    entities = retriever._extract_entities_enhanced(query)
    
    # Simular la búsqueda
    if entities['query_type'] == 'count':
        docs = retriever._handle_count_query_enhanced(query, entities)
    elif entities['names']:
        docs = retriever._search_by_speaker_enhanced(entities['names'])
    elif entities['topics']:
        docs = retriever._search_by_topic_enhanced(entities['topics'])
    elif entities['time_queries']:
        docs = retriever._search_by_time_period_enhanced(entities['time_queries'])
    elif entities['room_names']:
        docs = retriever._search_by_room_enhanced(entities['room_names'])
    else:
        docs = knowledge_base.vector_store.similarity_search(query, k=5)
    
    return {
        'query': query,
        'entities_extracted': entities,
        'documents_found': len(docs),
        'documents_preview': [
            {
                'session_name': doc.metadata.get('session_name', 'N/A'),
                'speaker_names': doc.metadata.get('speaker_names', 'N/A'),
                'start_time': doc.metadata.get('start_time', 'N/A')
            }
            for doc in docs[:3]
        ]
    }


def get_knowledge_base_stats(knowledge_base: EnhancedEventKnowledgeBase) -> Dict[str, Any]:
    """Obtiene estadísticas de la base de conocimiento (para debugging)"""
    return {
        'total_documents': len(knowledge_base.documents),
        'unique_speakers': len(knowledge_base.indices['by_speaker']),
        'speaker_variations': len(knowledge_base.indices['by_speaker_variations']),
        'unique_sessions': len(knowledge_base.indices['by_session']),
        'topics_indexed': len(knowledge_base.indices['by_topic_advanced']),
        'rooms_indexed': len(knowledge_base.indices['by_room']),
        'companies_indexed': len(knowledge_base.indices['by_company']),
        'time_periods': {
            period: len(docs) 
            for period, docs in knowledge_base.indices['by_time_period'].items()
        },
        'sample_speakers': list(knowledge_base.indices['by_speaker'].keys())[:10],
        'sample_topics': list(knowledge_base.indices['by_topic_advanced'].keys()),
        'sample_rooms': list(knowledge_base.indices['by_room'].keys())
    }


# Funciones de utilidad para testing específico
def test_speaker_search(speaker_name: str, knowledge_base: EnhancedEventKnowledgeBase) -> List[str]:
    """Prueba la búsqueda de un speaker específico"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    docs = retriever._search_by_speaker_enhanced([speaker_name])
    return [doc.metadata.get('session_name', 'N/A') for doc in docs]


def test_topic_search(topic: str, knowledge_base: EnhancedEventKnowledgeBase) -> List[str]:
    """Prueba la búsqueda de un tema específico"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    docs = retriever._search_by_topic_enhanced([topic])
    return [doc.metadata.get('session_name', 'N/A') for doc in docs]


def test_time_search(time_period: str, knowledge_base: EnhancedEventKnowledgeBase) -> List[str]:
    """Prueba la búsqueda por período de tiempo"""
    retriever = OptimizedEventRetriever(knowledge_base=knowledge_base)
    docs = retriever._search_by_time_period_enhanced([time_period])
    return [
        f"{doc.metadata.get('session_name', 'N/A')} - {doc.metadata.get('start_time', 'N/A')}"
        for doc in docs
    ]


# Configuración de logging específica para debugging
def enable_debug_logging():
    """Habilita logging detallado para debugging"""
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logger.info("Debug logging habilitado para query_helper")


def disable_debug_logging():
    """Deshabilita logging detallado"""
    logging.getLogger(__name__).setLevel(logging.WARNING)
    logger.info("Debug logging deshabilitado para query_helper")