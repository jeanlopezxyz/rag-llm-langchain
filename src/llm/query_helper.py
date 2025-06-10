from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Any
from vector_db.db_provider_factory import FAISS, DBFactory
from pydantic import ConfigDict, Field

############################
# LLM chain implementation - INTELIGENTE
############################

db_factory = DBFactory()

# INFORMACIÓN PARAMETRIZADA DESDE .ENV
EVENT_NAME = os.getenv("EVENT_NAME", "KCD Antigua Guatemala 2025")
EVENT_DATE = os.getenv("EVENT_DATE", "14 de junio de 2025")
EVENT_LOCATION = os.getenv("EVENT_LOCATION", "Centro de Convenciones Antigua, Guatemala")
EVENT_TIME = os.getenv("EVENT_TIME", "09:00 - 17:00")
ORGANIZATION = os.getenv("ORGANIZATION", "Cloud Native Community Guatemala")

# PROMPT ANTI-ALUCINACIÓN
smart_prompt_template = f"""Eres un asistente experto del {EVENT_NAME}.

INFORMACIÓN DEL EVENTO:
- Evento: {EVENT_NAME}
- Fecha: {EVENT_DATE}
- Ubicación: {EVENT_LOCATION}
- Horario: {EVENT_TIME}
- Salas: Salón Landívar, Salón El Obispo, Salón Don Pedro

INSTRUCCIONES CRÍTICAS:
1. USA SOLO la información del CONTEXTO proporcionado
2. NO INVENTES nombres de ponentes, horarios o sesiones
3. Si NO encuentras información específica, di "No encontré información sobre [lo que preguntaron]"
4. Para sesiones, usa EXACTAMENTE este formato:

🎯 **SESIÓN:** [nombre exacto de la sesión del contexto]
👤 **SPEAKER:** [nombre exacto del ponente del contexto]
⏰ **HORARIO:** [horario exacto del contexto]
🏢 **SALÓN:** [sala exacta del contexto]

5. Si preguntan por una sala específica, lista SOLO las sesiones de esa sala
6. NUNCA uses datos ficticios o ejemplos

CONTEXTO CON INFORMACIÓN REAL:
{{context}}

PREGUNTA: {{question}}

RESPUESTA (usar solo datos del contexto):"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=smart_prompt_template, 
    input_variables=["context", "question"]
)

def extract_session_info(content: str) -> dict:
    """Extraer información estructurada de una sesión."""
    session_info = {
        "session_name": "No especificado",
        "speaker": "No especificado", 
        "company": "",
        "start_time": "No especificado",
        "end_time": "",
        "room": "No especificado",
        "technologies": []
    }
    
    # Buscar patrones específicos en el contenido
    content_upper = content.upper()
    
    # Extraer nombre de la sesión (buscar después de "SESIÓN:" o patrones similares)
    session_patterns = [
        "SESIÓN: ",
        "SESSION: ",
        "CHARLA: ",
        "PRESENTACIÓN: ",
        "TALK: "
    ]
    
    for pattern in session_patterns:
        if pattern in content_upper:
            start_idx = content_upper.find(pattern) + len(pattern)
            # Buscar hasta el siguiente punto o hasta "TIPO DE SESIÓN"
            end_markers = [". TIPO", ". PONENTE", ". SPEAKER", ".", "\n"]
            end_idx = len(content)
            
            for marker in end_markers:
                marker_idx = content_upper.find(marker, start_idx)
                if marker_idx != -1 and marker_idx < end_idx:
                    end_idx = marker_idx
            
            session_name = content[start_idx:end_idx].strip()
            if session_name and len(session_name) > 3:
                session_info["session_name"] = session_name
                break
    
    # Extraer speaker/ponente
    speaker_patterns = [
        "PONENTE(S): ",
        "PONENTE: ",
        "SPEAKER(S): ",
        "SPEAKER: "
    ]
    
    for pattern in speaker_patterns:
        if pattern in content_upper:
            start_idx = content_upper.find(pattern) + len(pattern)
            # Buscar hasta el siguiente punto o campo
            end_markers = [". EMPRESA", ". FECHA", ". HORARIO", ".", "\n"]
            end_idx = len(content)
            
            for marker in end_markers:
                marker_idx = content_upper.find(marker, start_idx)
                if marker_idx != -1 and marker_idx < end_idx:
                    end_idx = marker_idx
            
            speaker_text = content[start_idx:end_idx].strip()
            if speaker_text:
                # Extraer empresa entre paréntesis
                if '(' in speaker_text and ')' in speaker_text:
                    parts = speaker_text.split('(')
                    session_info["speaker"] = parts[0].strip()
                    session_info["company"] = parts[1].replace(')', '').strip()
                else:
                    session_info["speaker"] = speaker_text
                break
    
    # Extraer horario
    if "HORARIO: DE " in content_upper:
        horario_start = content_upper.find("HORARIO: DE ") + len("HORARIO: DE ")
        horario_text = content[horario_start:horario_start+50]  # Tomar primeros 50 chars
        
        if " A " in horario_text.upper():
            times = horario_text.split(" A ")[:2]  # Solo tomar los primeros 2 elementos
            session_info["start_time"] = times[0].strip().split(".")[0]  # Remover puntos extras
            if len(times) > 1:
                session_info["end_time"] = times[1].strip().split(".")[0]
    
    # Extraer sala
    if "SALA: " in content_upper:
        sala_start = content_upper.find("SALA: ") + len("SALA: ")
        sala_text = content[sala_start:sala_start+30].split(".")[0].strip()
        session_info["room"] = sala_text
    
    # Extraer tecnologías
    if "TECNOLOGÍAS Y TEMAS: " in content_upper:
        tech_start = content_upper.find("TECNOLOGÍAS Y TEMAS: ") + len("TECNOLOGÍAS Y TEMAS: ")
        tech_text = content[tech_start:tech_start+100].split(".")[0].strip()
        if tech_text:
            session_info["technologies"] = [t.strip() for t in tech_text.split(',')]
    
    return session_info

def format_session_response(documents: List[Document], query: str = "") -> str:
    """Formatear respuesta con estructura clara y datos REALES únicamente."""
    formatted_sessions = []
    query_lower = query.lower()
    
    # Detectar si es una consulta específica por nombre o sala
    query_words = query.split()
    specific_names = [word for word in query_words if word.istitle() and len(word) > 2]
    
    # Detectar sala específica
    target_room = None
    if "landívar" in query_lower or "landivar" in query_lower:
        target_room = "landívar"
    elif "obispo" in query_lower:
        target_room = "obispo"
    elif "pedro" in query_lower:
        target_room = "pedro"
    
    print(f"🔍 Filtros detectados - Nombres: {specific_names}, Sala: {target_room}")
    
    for doc in documents:
        if doc.metadata.get("type") == "informacion_general":
            continue  # Saltar info general del evento
            
        session_info = extract_session_info(doc.page_content)
        
        # VALIDACIÓN: Solo procesar si tenemos datos REALES
        if (session_info["session_name"] == "No especificado" and 
            session_info["speaker"] == "No especificado"):
            print(f"⚠️ Saltando documento sin datos válidos")
            continue
        
        # FILTRADO POR SPEAKER ESPECÍFICO
        if specific_names:
            speaker_matches = False
            speaker_text = session_info["speaker"].lower()
            
            # Verificar si TODOS los nombres específicos están en el speaker
            for name in specific_names:
                if name.lower() in speaker_text:
                    speaker_matches = True
                else:
                    speaker_matches = False
                    break
            
            if not speaker_matches:
                continue
        
        # FILTRADO POR SALA ESPECÍFICA
        if target_room:
            content_lower = doc.page_content.lower()
            room_code_mapping = {
                "landívar": "room-1",
                "obispo": "room-2", 
                "pedro": "room-3"
            }
            
            expected_room_code = room_code_mapping.get(target_room, "")
            if expected_room_code and expected_room_code not in content_lower:
                # También verificar por nombre de sala
                if target_room not in content_lower:
                    print(f"🚫 Sesión no pertenece a {target_room}: {session_info['session_name'][:30]}...")
                    continue
        
        # VERIFICAR QUE TENEMOS DATOS REALES (no ficticios)
        if ("[Nombre" in session_info["session_name"] or 
            "ficticios" in session_info["speaker"].lower() or
            "ejemplo" in session_info["speaker"].lower()):
            print(f"🚫 Datos ficticios detectados, saltando...")
            continue
        
        # Formatear la sesión con datos reales
        formatted_session = f"""🎯 **SESIÓN:** {session_info["session_name"]}
👤 **SPEAKER:** {session_info["speaker"]}"""
        
        if session_info["company"]:
            formatted_session += f" ({session_info['company']})"
        
        formatted_session += f"""
⏰ **HORARIO:** {session_info["start_time"]}"""
        
        if session_info["end_time"]:
            formatted_session += f" - {session_info['end_time']}"
        
        # Convertir códigos de sala a nombres reales
        room_name = session_info["room"]
        if "ROOM-1" in room_name.upper() or "LANDÍVAR" in room_name.upper():
            room_name = "Salón Landívar"
        elif "ROOM-2" in room_name.upper() or "OBISPO" in room_name.upper():
            room_name = "Salón El Obispo"
        elif "ROOM-3" in room_name.upper() or "PEDRO" in room_name.upper():
            room_name = "Salón Don Pedro"
        
        formatted_session += f"""
🏢 **SALÓN:** {room_name}"""
        
        if session_info["technologies"]:
            formatted_session += f"""
🔧 **TECNOLOGÍAS:** {', '.join(session_info['technologies'])}"""
        
        formatted_sessions.append(formatted_session)
        print(f"✅ Sesión válida añadida: {session_info['session_name'][:40]}...")
    
    print(f"📋 Total sesiones formateadas: {len(formatted_sessions)}")
    
    # LIMITAR RESULTADOS según el tipo de consulta
    if target_room:
        return formatted_sessions[:8]  # Más sesiones para consultas por sala
    else:
        return formatted_sessions[:3]  # Menos para consultas generales

def analyze_query_intent(question: str) -> dict:
    """Analizar la intención de la consulta para mejor búsqueda."""
    q = question.lower()
    
    intent = {
        "type": "general",
        "entities": [],
        "time_filter": None,
        "needs_search": True
    }
    
    # Detectar nombres de personas (capitalización)
    words = question.split()
    potential_names = [word for word in words if word.istitle() and len(word) > 2]
    if potential_names:
        intent["entities"] = potential_names
        intent["type"] = "speaker_query"
    
    # Detectar consultas temporales
    if any(word in q for word in ["mañana", "morning", "am"]):
        intent["time_filter"] = "morning"
        intent["type"] = "schedule_query"
    elif any(word in q for word in ["tarde", "afternoon", "pm"]):
        intent["time_filter"] = "afternoon"
        intent["type"] = "schedule_query"
    
    # Detectar consultas sobre ubicación/salas
    if any(word in q for word in ["sala", "salón", "room", "dónde"]):
        intent["type"] = "location_query"
    
    # Detectar consultas sobre tecnologías
    tech_keywords = ["kubernetes", "k8s", "docker", "devops", "ci/cd", "helm", "security", "ia", "ai"]
    mentioned_tech = [tech for tech in tech_keywords if tech in q]
    if mentioned_tech:
        intent["entities"].extend(mentioned_tech)
        intent["type"] = "technology_query"
    
    return intent

def get_direct_answer(question: str) -> str:
    """Respuestas directas mejoradas."""
    q = question.lower()
    
    # Respuestas básicas del evento
    if any(word in q for word in ["dónde", "donde", "ubicación"]) and not any(word in q for word in ["sala", "salón"]):
        return f"📍 El {EVENT_NAME} será en el **{EVENT_LOCATION}** el {EVENT_DATE}."
    
    elif any(word in q for word in ["cuándo", "cuando", "fecha"]) and "hora" not in q:
        return f"📅 El {EVENT_NAME} es el **{EVENT_DATE}** de {EVENT_TIME}."
    
    elif any(word in q for word in ["horario", "hora"]) and not any(word in q for word in ["mañana", "tarde"]):
        return f"⏰ El horario del {EVENT_NAME} es **{EVENT_TIME}** el {EVENT_DATE}."
    
    # Información sobre salas
    elif any(word in q for word in ["sala", "salón", "room"]) and not any(name in q for name in ["jean", "paul", "sergio"]):
        return f"""🏢 **Salas del {EVENT_NAME}:**

📍 **Salón Landívar** (Sala principal)
📍 **Salón El Obispo** (Sala secundaria)  
📍 **Salón Don Pedro** (Sala terciaria)

Todas ubicadas en el {EVENT_LOCATION}."""
    
    elif any(word in q for word in ["organiza", "organizador"]):
        return f"🏢 El {EVENT_NAME} es organizado por **{ORGANIZATION}**."
    
    return None

class SmartKCDRetriever(BaseRetriever):
    """Retriever inteligente que entiende mejor las consultas."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_retriever: Any = Field(description="El retriever base")
    
    def __init__(self, base_retriever, **kwargs):
        super().__init__(base_retriever=base_retriever, **kwargs)
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.get_relevant_documents(query)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Búsqueda inteligente basada en intención."""
        print(f"🔍 Consultando: '{query}'")
        
        # Analizar la intención de la consulta
        intent = analyze_query_intent(query)
        print(f"🧠 Intención detectada: {intent['type']}")
        if intent['entities']:
            print(f"🏷️ Entidades: {intent['entities']}")
        
        try:
            docs = []
            
            # Estrategia de búsqueda según el tipo de consulta
            if intent["type"] == "speaker_query":
                docs = self.search_by_speaker(intent["entities"])
            elif intent["type"] == "schedule_query":
                docs = self.search_by_schedule(query, intent["time_filter"])
            elif intent["type"] == "technology_query":
                docs = self.search_by_technology(intent["entities"])
            elif intent["type"] == "location_query":
                docs = self.search_by_location(query)
            else:
                docs = self.general_search(query)
            
            # Añadir documento de información general del evento
            event_doc = self.create_event_info_doc()
            
            # Combinar resultados, priorizando los específicos
            final_docs = [event_doc] + docs[:4]  # Máximo 5 documentos total
            
            print(f"📄 {len(final_docs)} documentos preparados")
            return final_docs
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return [self.create_event_info_doc()]
    
    def search_by_speaker(self, names: List[str]) -> List[Document]:
        """Búsqueda específica y precisa por nombres de speakers."""
        docs = []
        
        # Crear consultas más específicas
        search_queries = []
        
        # Si hay múltiples nombres, buscar como frase completa
        if len(names) >= 2:
            full_name = " ".join(names)
            search_queries.append(f'ponente "{full_name}"')
            search_queries.append(f'speaker "{full_name}"')
            search_queries.append(full_name)
        
        # También buscar nombres individuales como fallback
        for name in names:
            search_queries.append(f"ponente {name}")
            search_queries.append(f"speaker {name}")
        
        print(f"🔍 Búsquedas por speaker: {search_queries}")
        
        for query in search_queries:
            try:
                if hasattr(self.base_retriever, 'vectorstore'):
                    speaker_docs = self.base_retriever.vectorstore.similarity_search(query, k=2)
                else:
                    speaker_docs = self.base_retriever.get_relevant_documents(query)[:2]
                
                # Filtrar documentos que realmente contengan el nombre
                filtered_docs = []
                for doc in speaker_docs:
                    content_lower = doc.page_content.lower()
                    # Verificar que el contenido realmente contiene los nombres buscados
                    if len(names) >= 2:
                        full_name_lower = " ".join(names).lower()
                        if full_name_lower in content_lower:
                            filtered_docs.append(doc)
                    else:
                        if names[0].lower() in content_lower:
                            filtered_docs.append(doc)
                
                docs.extend(filtered_docs)
                print(f"   👤 {len(filtered_docs)} documentos relevantes para '{query}'")
                
                # Si encontramos resultados específicos, no seguir buscando
                if filtered_docs:
                    break
                    
            except Exception as e:
                print(f"   ⚠️ Error buscando speaker '{query}': {e}")
        
        # Eliminar duplicados
        unique_docs = []
        seen_content = set()
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs[:2]  # Máximo 2 documentos para speakers específicos
    
    def search_by_schedule(self, query: str, time_filter: str) -> List[Document]:
        """Búsqueda por horarios."""
        search_terms = []
        
        if time_filter == "morning":
            search_terms = ["09:00", "10:00", "11:00", "mañana"]
        elif time_filter == "afternoon":
            search_terms = ["14:00", "15:00", "16:00", "tarde"]
        else:
            search_terms = ["horario", "hora"]
        
        docs = []
        for term in search_terms:
            try:
                if hasattr(self.base_retriever, 'vectorstore'):
                    time_docs = self.base_retriever.vectorstore.similarity_search(term, k=2)
                else:
                    time_docs = self.base_retriever.get_relevant_documents(term)[:2]
                
                docs.extend(time_docs)
                print(f"   ⏰ {len(time_docs)} documentos para '{term}'")
            except Exception as e:
                print(f"   ⚠️ Error buscando horario '{term}': {e}")
        
        return docs
    
    def search_by_technology(self, technologies: List[str]) -> List[Document]:
        """Búsqueda por tecnologías específicas."""
        docs = []
        for tech in technologies:
            try:
                if hasattr(self.base_retriever, 'vectorstore'):
                    tech_docs = self.base_retriever.vectorstore.similarity_search(tech, k=3)
                else:
                    tech_docs = self.base_retriever.get_relevant_documents(tech)[:3]
                
                docs.extend(tech_docs)
                print(f"   🔧 {len(tech_docs)} documentos para tecnología '{tech}'")
            except Exception as e:
                print(f"   ⚠️ Error buscando tecnología '{tech}': {e}")
        
        return docs
    
    def search_by_location(self, query: str) -> List[Document]:
        """Búsqueda mejorada por ubicación/salas."""
        q_lower = query.lower()
        docs = []
        
        # Detectar qué sala específica se está buscando
        target_room = None
        if "landívar" in q_lower or "landivar" in q_lower:
            target_room = "ROOM-1"
            search_terms = ["landívar", "landivar", "room-1", "sala 1"]
        elif "obispo" in q_lower:
            target_room = "ROOM-2"
            search_terms = ["obispo", "room-2", "sala 2"]
        elif "pedro" in q_lower:
            target_room = "ROOM-3"
            search_terms = ["pedro", "room-3", "sala 3"]
        else:
            # Búsqueda general por salas
            search_terms = ["sala", "salón", "room", "landívar", "obispo", "pedro"]
        
        print(f"🏢 Buscando en sala específica: {target_room or 'general'}")
        
        # Buscar documentos
        for term in search_terms[:3]:  # Limitar búsquedas
            try:
                if hasattr(self.base_retriever, 'vectorstore'):
                    loc_docs = self.base_retriever.vectorstore.similarity_search(term, k=6)
                else:
                    loc_docs = self.base_retriever.get_relevant_documents(term)[:6]
                
                # Filtrar por sala específica si se detectó una
                if target_room:
                    filtered_docs = []
                    for doc in loc_docs:
                        if target_room in doc.page_content.upper():
                            filtered_docs.append(doc)
                    docs.extend(filtered_docs)
                    print(f"   🏢 {len(filtered_docs)} documentos para sala {target_room}")
                else:
                    docs.extend(loc_docs)
                    print(f"   🏢 {len(loc_docs)} documentos para ubicación '{term}'")
                
                # Si encontramos suficientes documentos específicos, parar
                if target_room and len(docs) >= 5:
                    break
                    
            except Exception as e:
                print(f"   ⚠️ Error buscando ubicación '{term}': {e}")
        
        # Eliminar duplicados
        unique_docs = []
        seen_content = set()
        for doc in docs:
            content_hash = hash(doc.page_content[:100])  # Hash de los primeros 100 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        print(f"🏢 Total documentos únicos encontrados: {len(unique_docs)}")
        return unique_docs[:8]  # Máximo 8 para múltiples sesiones por sala
    
    def general_search(self, query: str) -> List[Document]:
        """Búsqueda general mejorada."""
        try:
            if hasattr(self.base_retriever, 'vectorstore'):
                docs = self.base_retriever.vectorstore.similarity_search(query, k=4)
            else:
                docs = self.base_retriever.get_relevant_documents(query)[:4]
            
            print(f"   🔍 {len(docs)} documentos en búsqueda general")
            return docs
        except Exception as e:
            print(f"   ⚠️ Error en búsqueda general: {e}")
            return []
    
    def create_event_info_doc(self) -> Document:
        """Crear documento con información completa del evento."""
        return Document(
            page_content=f"""
{EVENT_NAME} - Información Completa:

UBICACIÓN: {EVENT_LOCATION}
FECHA: {EVENT_DATE}
HORARIO: {EVENT_TIME}
ORGANIZADOR: {ORGANIZATION}

SALAS DISPONIBLES:
- Salón Landívar (Sala principal, 200 personas)
- Salón El Obispo (Sala secundaria, 200 personas)  
- Salón Don Pedro (Sala terciaria, 200 personas)

HORARIOS:
- Mañana: 09:00 - 12:00
- Tarde: 14:00 - 17:00

MODALIDAD: Presencial y gratuito con registro previo
TEMÁTICA: Cloud Native, Kubernetes, DevOps, Contenedores
            """.strip(),
            metadata={
                "source": "evento_info_completa",
                "type": "informacion_general",
                "event_name": EVENT_NAME
            }
        )

def get_qa_chain(llm):
    """Crear cadena QA inteligente."""
    db_type = os.getenv("DB_TYPE", "PGVECTOR")
    
    try:
        print(f"🗃️ Inicializando DB: {db_type}")
        base_retriever = db_factory.get_retriever(db_type)
        print(f"✅ {db_type} configurado")
        
        # Configuración optimizada
        if hasattr(base_retriever, 'vectorstore'):
            retriever = base_retriever.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Aumentado para búsquedas específicas
            )
        else:
            retriever = base_retriever
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        raise ValueError(f"Error en DB: {e}")

    # Usar retriever inteligente
    smart_retriever = SmartKCDRetriever(base_retriever=retriever)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=smart_retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "verbose": False
        },
        return_source_documents=True,
        input_key="query",
        output_key="result"
    )

def format_kcd_query(user_question):
    """Formatear consultas con análisis previo."""
    # Intentar respuesta directa
    direct_answer = get_direct_answer(user_question)
    if direct_answer:
        return direct_answer
    
    # Analizar intención para mejor contexto
    intent = analyze_query_intent(user_question)
    
    if intent["type"] == "speaker_query" and intent["entities"]:
        return f"Consulta sobre speaker(s): {', '.join(intent['entities'])} en {EVENT_NAME}"
    elif intent["type"] == "schedule_query":
        return f"Consulta sobre horarios del {EVENT_NAME}: {user_question}"
    elif intent["type"] == "technology_query" and intent["entities"]:
        return f"Consulta sobre tecnologías en {EVENT_NAME}: {user_question}"
    else:
        return user_question

# Exportar funciones principales
__all__ = [
    'get_qa_chain',
    'format_kcd_query',
    'get_direct_answer',
    'analyze_query_intent'
]