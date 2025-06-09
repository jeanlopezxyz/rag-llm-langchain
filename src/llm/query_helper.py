from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
from vector_db.db_provider_factory import FAISS, DBFactory

############################
# LLM chain implementation #
############################

db_factory = DBFactory()

# PROMPT ESPECIALIZADO PARA CONSULTAS DE EVENTOS
prompt_template = """
### [INST]
Eres un asistente experto en eventos, charlas y conferencias. Tu función es responder preguntas específicas sobre:
- Horarios de charlas y eventos
- Información de speakers y ponentes
- Ubicaciones y salas de eventos
- Temas y contenidos de las charlas
- Agenda y programación
- Logística del evento

CONTEXTO DISPONIBLE:
{context}

INSTRUCCIONES:
- Responde de forma directa y concisa
- Si preguntan por horarios, proporciona fechas y horas exactas
- Si preguntan por ubicaciones, sé específico sobre salas/auditorios
- Si preguntan por speakers, incluye su información relevante
- Si preguntan por temas, relaciona las charlas correspondientes
- Para agendas, organiza por horario y evita conflictos
- Si no tienes la información exacta, dilo claramente
- Usa formato claro con bullets o listas cuando sea apropiado

EJEMPLOS DE RESPUESTAS:
- "Hoy hay 3 charlas programadas: ..."
- "Las charlas de IA son: [lista con horarios]"
- "El Dr. García tiene 2 charlas: ..."
- "El evento se realiza en el Centro de Convenciones Madrid"

PREGUNTA DEL USUARIO:
{question}

Responde de forma útil y directa basándote en el contexto disponible.
[/INST]
"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_qa_chain(llm):
    """
    Crea y retorna una cadena de QA optimizada para consultas sobre eventos.
    
    Args:
        llm: Instancia del modelo de lenguaje a utilizar
        
    Returns:
        RetrievalQA: Cadena configurada para consultas de eventos
    """
    try:
        db_type = os.getenv("DB_TYPE") if os.getenv("DB_TYPE") else "FAISS"
        if db_type is None:
            raise ValueError("DB_TYPE no está especificado")
        
        print(f"🗃️ Inicializando base de datos vectorial: {db_type}")
        retriever = db_factory.get_retriever(db_type)
        print(f"✅ Retriever {db_type} configurado correctamente")
        
    except Exception as e:
        print(f"⚠️ Error configurando {db_type}: {e}")
        print(f"🔄 Fallback a FAISS")
        retriever = db_factory.get_retriever(FAISS)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "verbose": False
        },
        return_source_documents=True,
        input_key="query",
        output_key="result"
    )

# ==========================================
# FUNCIONES AUXILIARES PARA CONSULTAS
# ==========================================

def format_event_query(user_question):
    """
    Formatea la pregunta del usuario para mejor procesamiento RAG.
    
    Args:
        user_question (str): Pregunta original del usuario
        
    Returns:
        str: Pregunta formateada y optimizada
    """
    
    # Palabras clave para diferentes tipos de consultas
    time_keywords = ["hoy", "mañana", "día", "fecha", "horario", "hora", "cuándo"]
    speaker_keywords = ["speaker", "ponente", "conferenciante", "quién", "dr.", "dra."]
    location_keywords = ["dónde", "ubicación", "sala", "auditorio", "lugar", "local"]
    topic_keywords = ["tema", "sobre", "relacionado", "inteligencia artificial", "ia", "devops", "machine learning"]
    agenda_keywords = ["agenda", "cronograma", "programación", "horarios", "cronograma"]
    
    # Clasificar tipo de consulta
    query_type = "general"
    user_lower = user_question.lower()
    
    if any(keyword in user_lower for keyword in time_keywords):
        query_type = "horario"
    elif any(keyword in user_lower for keyword in speaker_keywords):
        query_type = "speaker"
    elif any(keyword in user_lower for keyword in location_keywords):
        query_type = "ubicacion"
    elif any(keyword in user_lower for keyword in topic_keywords):
        query_type = "tema"
    elif any(keyword in user_lower for keyword in agenda_keywords):
        query_type = "agenda"
    
    # Agregar contexto según el tipo de consulta
    context_prefixes = {
        "horario": "Consulta sobre horarios y fechas de eventos: ",
        "speaker": "Consulta sobre speakers y ponentes: ",
        "ubicacion": "Consulta sobre ubicaciones y lugares: ",
        "tema": "Consulta sobre temas y contenidos: ",
        "agenda": "Consulta sobre agenda y programación: ",
        "general": "Consulta general sobre eventos: "
    }
    
    formatted_query = context_prefixes[query_type] + user_question
    
    return formatted_query

def extract_key_entities(question):
    """
    Extrae entidades clave de la pregunta para mejor búsqueda.
    
    Args:
        question (str): Pregunta del usuario
        
    Returns:
        dict: Entidades extraídas (fechas, nombres, temas, etc.)
    """
    
    entities = {
        "dates": [],
        "speakers": [],
        "topics": [],
        "locations": []
    }
    
    question_lower = question.lower()
    
    # Detectar fechas comunes
    date_patterns = ["hoy", "mañana", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    for pattern in date_patterns:
        if pattern in question_lower:
            entities["dates"].append(pattern)
    
    # Detectar nombres de speakers (patrones comunes)
    if "dr." in question_lower or "dra." in question_lower:
        words = question.split()
        for i, word in enumerate(words):
            if word.lower() in ["dr.", "dra."] and i + 1 < len(words):
                entities["speakers"].append(f"{word} {words[i+1]}")
    
    # Detectar temas tecnológicos
    tech_topics = [
        "inteligencia artificial", "ia", "machine learning", "ml", 
        "devops", "cloud", "kubernetes", "docker", "python", 
        "javascript", "react", "nodejs", "data science"
    ]
    
    for topic in tech_topics:
        if topic in question_lower:
            entities["topics"].append(topic)
    
    return entities

def suggest_related_questions(user_question):
    """
    Sugiere preguntas relacionadas basadas en la consulta del usuario.
    
    Args:
        user_question (str): Pregunta del usuario
        
    Returns:
        list: Lista de preguntas sugeridas
    """
    
    question_lower = user_question.lower()
    suggestions = []
    
    if "horario" in question_lower or "hora" in question_lower:
        suggestions = [
            "¿Qué charlas hay disponibles mañana?",
            "¿Cuál es la agenda completa del evento?",
            "¿A qué hora termina el evento?"
        ]
    elif "speaker" in question_lower or "ponente" in question_lower:
        suggestions = [
            "¿Cuántos speakers participan en total?",
            "¿Qué experiencia tiene este speaker?",
            "¿En qué otras charlas participa?"
        ]
    elif "tema" in question_lower or "sobre" in question_lower:
        suggestions = [
            "¿Qué otros temas se cubren en el evento?",
            "¿Hay charlas relacionadas disponibles?",
            "¿Cuál es el nivel técnico de estas charlas?"
        ]
    else:
        suggestions = [
            "¿Cuántas charlas hay en total?",
            "¿Dónde se realiza el evento?",
            "¿Cuál es el horario general del evento?"
        ]
    
    return suggestions

def validate_time_conflicts(events_list):
    """
    Valida conflictos de horario en una lista de eventos.
    
    Args:
        events_list (list): Lista de eventos con horarios
        
    Returns:
        dict: Información sobre conflictos encontrados
    """
    
    conflicts = {
        "has_conflicts": False,
        "conflict_details": [],
        "suggested_resolution": []
    }
    
    # Esta función se puede expandir para validar horarios reales
    # cuando se tengan datos estructurados de eventos
    
    return conflicts

# ==========================================
# CONFIGURACIÓN ESPECÍFICA PARA EVENTOS
# ==========================================

def get_event_specific_params():
    """
    Retorna parámetros específicos para consultas de eventos.
    
    Returns:
        dict: Parámetros optimizados para eventos
    """
    
    return {
        "temperature": 0.3,  # Respuestas más precisas para datos factuales
        "max_tokens": 500,   # Respuestas concisas
        "top_p": 0.9
    }

def create_event_context(question_type):
    """
    Crea contexto específico según el tipo de pregunta sobre eventos.
    
    Args:
        question_type (str): Tipo de pregunta (horario, speaker, tema, etc.)
        
    Returns:
        str: Contexto adicional para el LLM
    """
    
    contexts = {
        "horario": "Enfócate en proporcionar horarios exactos, fechas específicas y duración de las charlas.",
        "speaker": "Proporciona información detallada sobre la experiencia y expertise del speaker.",
        "tema": "Explica el contenido de las charlas y su relevancia técnica.",
        "ubicacion": "Sé específico sobre salas, auditorios y cómo llegar al lugar.",
        "agenda": "Organiza la información cronológicamente y evita conflictos de horario."
    }
    
    return contexts.get(question_type, "Proporciona información precisa y útil sobre el evento.")

# Función principal mejorada para chatbot
def enhance_query_for_chatbot(user_question):
    """
    Mejora la query del usuario para mejor experiencia de chatbot.
    
    Args:
        user_question (str): Pregunta original del usuario
        
    Returns:
        dict: Query mejorada con contexto adicional
    """
    
    formatted_query = format_event_query(user_question)
    entities = extract_key_entities(user_question)
    suggestions = suggest_related_questions(user_question)
    
    return {
        "formatted_query": formatted_query,
        "entities": entities,
        "suggestions": suggestions,
        "original_question": user_question
    }