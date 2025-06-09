from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
from vector_db.db_provider_factory import FAISS, DBFactory

############################
# LLM chain implementation #
############################

db_factory = DBFactory()

# MODIFICADO: Prompt template completamente rediseñado para eventos de charlas
prompt_template = """
### [INST]
Instrucciones para Generación de Propuestas de Eventos con Speakers:

Eres un asistente experto en organización de eventos y coordinación de speakers profesionales. Tu función es crear propuestas integrales, detalladas y profesionales para eventos de charlas, conferencias y presentaciones.

CONTEXTO Y CONOCIMIENTO BASE:
{context}

ESTRUCTURA REQUERIDA PARA LA PROPUESTA:
Genera una propuesta profesional en formato markdown que incluya OBLIGATORIAMENTE estas 8 secciones:

## 1. 🎯 **RESUMEN EJECUTIVO**
- Objetivo del evento y propuesta de valor
- Resumen de la charla en 2-3 líneas clave
- Audiencia objetivo y número estimado de asistentes

## 2. 👤 **PERFIL DEL SPEAKER**
- Biografía profesional del speaker (150-200 palabras)
- Experiencia relevante y credenciales
- Especializaciones y áreas de expertise
- Eventos previos destacados
- Contacto y redes sociales

## 3. 📅 **DETALLES DEL EVENTO**
- Nombre completo del evento
- Fecha, horario exacto y duración
- Ubicación específica (sala, edificio, dirección)
- Formato (presencial, virtual, híbrido)
- Capacidad y tipo de audiencia

## 4. 💡 **CONTENIDO Y AGENDA DE LA SESIÓN**
- Título de la presentación
- Objetivos de aprendizaje (3-5 puntos específicos)
- Agenda detallada con timing
- Metodología (presentación, demo, Q&A, workshop)
- Materiales y recursos que se proporcionarán

## 5. 🛠️ **REQUERIMIENTOS TÉCNICOS Y LOGÍSTICA**
- Equipamiento audiovisual necesario
- Requerimientos de conectividad (WiFi, streaming)
- Setup del escenario y disposición
- Materiales promocionales
- Necesidades especiales del speaker

## 6. 🎯 **AUDIENCIA Y EXPECTATIVAS**
- Perfil detallado de la audiencia objetivo
- Nivel técnico requerido (básico, intermedio, avanzado)
- Prerrequisitos de conocimiento
- Expectativas y resultados esperados
- Métrica de éxito del evento

## 7. 📢 **MARKETING Y PROMOCIÓN**
- Propuesta de copy para promoción
- Canales de difusión recomendados
- Timeline de marketing pre-evento
- Materiales gráficos sugeridos
- Estrategia de redes sociales

## 8. 🔄 **SEGUIMIENTO Y ACTIVIDADES POST-EVENTO**
- Actividades de networking programadas
- Materiales de seguimiento para asistentes
- Encuestas de satisfacción
- Grabación y distribución de contenido
- Próximos pasos y acciones recomendadas

DIRECTRICES DE CALIDAD:
- Usa un tono profesional pero accesible
- Incluye detalles específicos y accionables
- Cada sección debe tener 3-5 puntos bien desarrollados
- Utiliza emojis y formato markdown para mejor legibilidad
- La propuesta debe ser de 800-1200 palabras mínimo
- Personaliza el contenido según el tema y audiencia específica
- Incluye consideraciones de accesibilidad y diversidad

PREGUNTA/SOLICITUD:
{question}

Genera una propuesta integral siguiendo exactamente la estructura requerida, adaptando todo el contenido al evento específico mencionado en la pregunta.

[/INST]
"""

# MODIFICADO: Prompt template específico para eventos
QA_CHAIN_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_qa_chain(llm):
    """
    Crea y retorna una cadena de QA optimizada para generar propuestas de eventos.
    
    Args:
        llm: Instancia del modelo de lenguaje a utilizar
        
    Returns:
        RetrievalQA: Cadena configurada para propuestas de eventos
    """
    try:
        # MODIFICADO: Intenta obtener el tipo de DB desde variables de entorno
        db_type = os.getenv("DB_TYPE") if os.getenv("DB_TYPE") else "FAISS"
        if db_type is None:
            raise ValueError("DB_TYPE no está especificado")
        
        print(f"🗃️ Inicializando base de datos vectorial: {db_type}")
        retriever = db_factory.get_retriever(db_type)
        
        print(f"✅ Retriever {db_type} configurado correctamente")
        
    except Exception as e:
        print(f"⚠️ Error configurando {db_type}: {e}")
        print(f"🔄 Fallback a FAISS - Las propuestas se generarán sin contexto RAG adicional")
        
        # Fallback a FAISS como opción segura
        retriever = db_factory.get_retriever(FAISS)

    # MODIFICADO: Configuración optimizada para propuestas de eventos
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # Usar 'stuff' para mejor control del prompt
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "verbose": True  # Para debugging en desarrollo
        },
        return_source_documents=True,  # Incluir fuentes para transparencia
        input_key="query",  # Clave de entrada para la pregunta
        output_key="result"  # Clave de salida para el resultado
    )


def format_event_query(event_name, speaker_name, date, time_slot, location, topic, additional_info=""):
    """
    Formatea la información del evento en una query estructurada para el LLM.
    
    Args:
        event_name (str): Nombre del evento
        speaker_name (str): Nombre del speaker
        date (str): Fecha del evento
        time_slot (str): Horario del evento
        location (str): Ubicación del evento
        topic (str): Tema principal
        additional_info (str): Información adicional opcional
        
    Returns:
        str: Query formateada para el LLM
    """
    
    # MODIFICADO: Query estructurada específica para eventos
    formatted_query = f"""
SOLICITUD DE PROPUESTA PARA EVENTO DE CHARLA

📋 INFORMACIÓN DEL EVENTO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Evento: {event_name}
👤 Speaker: {speaker_name}
📅 Fecha: {date}
🕐 Horario: {time_slot}
📍 Ubicación: {location}
💡 Tema Principal: {topic}
"""
    
    if additional_info and additional_info.strip():
        formatted_query += f"📝 Información Adicional: {additional_info}\n"
    
    formatted_query += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 OBJETIVO:
Generar una propuesta profesional integral que incluya:
• Perfil detallado del speaker y su expertise
• Agenda completa de la sesión con objetivos de aprendizaje
• Requerimientos técnicos y logísticos específicos
• Estrategia de promoción y marketing del evento
• Plan de seguimiento post-evento
• Consideraciones de audiencia y expectativas

La propuesta debe ser detallada, profesional y específicamente adaptada a este evento de charla.
"""
    
    return formatted_query


def create_speaker_context(speaker_name, topic):
    """
    Crea contexto adicional sobre speakers y temas comunes.
    
    Args:
        speaker_name (str): Nombre del speaker
        topic (str): Tema de la charla
        
    Returns:
        str: Contexto adicional para el LLM
    """
    
    # Mapeo de temas a contextos especializados
    topic_contexts = {
        "inteligencia artificial": "contexto de IA, machine learning, deep learning, aplicaciones prácticas",
        "ia": "contexto de IA, machine learning, deep learning, aplicaciones prácticas",
        "machine learning": "algoritmos ML, casos de uso, herramientas, mejores prácticas",
        "desarrollo": "metodologías de desarrollo, frameworks, arquitectura de software",
        "devops": "CI/CD, containerización, automatización, monitoreo",
        "cloud": "servicios en la nube, arquitectura cloud, migración, seguridad",
        "ciberseguridad": "amenazas, protección, compliance, mejores prácticas",
        "blockchain": "tecnología distribuida, casos de uso, implementación",
        "frontend": "desarrollo web, frameworks JS, UX/UI, performance",
        "backend": "arquitectura de servicios, bases de datos, APIs",
        "mobile": "desarrollo móvil, apps nativas, cross-platform"
    }
    
    # Buscar contexto relevante basado en palabras clave del tema
    relevant_context = "desarrollo de software y tecnología"
    topic_lower = topic.lower()
    
    for keyword, context in topic_contexts.items():
        if keyword in topic_lower:
            relevant_context = context
            break
    
    context = f"""
CONTEXTO PARA LA PROPUESTA:

Speaker: {speaker_name}
- Se asume experiencia profesional en {relevant_context}
- Credenciales y trayectoria relevante al tema
- Capacidad de presentación a audiencias técnicas y no técnicas

Tema: {topic}
- Enfoque en aplicaciones prácticas y casos de uso reales
- Contenido actualizado con tendencias y mejores prácticas
- Adaptado al nivel de la audiencia objetivo

Consideraciones del Evento:
- Formato profesional y educativo
- Interacción con audiencia (Q&A, networking)
- Materiales de apoyo y seguimiento
- Métricas de éxito y satisfacción
"""
    
    return context


def validate_event_data(event_name, speaker_name, topic):
    """
    Valida que los datos del evento sean suficientes para generar una buena propuesta.
    
    Args:
        event_name (str): Nombre del evento
        speaker_name (str): Nombre del speaker  
        topic (str): Tema principal
        
    Returns:
        tuple: (is_valid, error_message)
    """
    
    errors = []
    
    if not event_name or len(event_name.strip()) < 3:
        errors.append("El nombre del evento debe tener al menos 3 caracteres")
    
    if not speaker_name or len(speaker_name.strip()) < 2:
        errors.append("El nombre del speaker debe tener al menos 2 caracteres")
    
    if not topic or len(topic.strip()) < 5:
        errors.append("El tema debe tener al menos 5 caracteres")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, ""


def enhance_query_with_context(base_query, event_name, speaker_name, topic):
    """
    Mejora la query base agregando contexto específico del dominio.
    
    Args:
        base_query (str): Query base formateada
        event_name (str): Nombre del evento
        speaker_name (str): Nombre del speaker
        topic (str): Tema principal
        
    Returns:
        str: Query mejorada con contexto adicional
    """
    
    # Agregar contexto específico
    speaker_context = create_speaker_context(speaker_name, topic)
    
    enhanced_query = f"{base_query}\n\n{speaker_context}"
    
    return enhanced_query


# MODIFICADO: Funciones de utilidad adicionales para eventos

def get_event_templates():
    """
    Retorna templates predefinidos para diferentes tipos de eventos.
    
    Returns:
        dict: Templates por tipo de evento
    """
    
    return {
        "conferencia_tech": {
            "duration": "45-60 minutos",
            "format": "Presentación + Q&A",
            "audience_size": "100-500 personas",
            "technical_level": "Intermedio-Avanzado"
        },
        "workshop": {
            "duration": "2-4 horas", 
            "format": "Hands-on + Ejercicios prácticos",
            "audience_size": "20-50 personas",
            "technical_level": "Variable según topic"
        },
        "keynote": {
            "duration": "30-45 minutos",
            "format": "Presentación inspiracional",
            "audience_size": "200-1000+ personas", 
            "technical_level": "General audience"
        },
        "panel": {
            "duration": "60-90 minutos",
            "format": "Discusión moderada + Q&A",
            "audience_size": "50-300 personas",
            "technical_level": "Intermedio"
        }
    }


def suggest_event_improvements(event_data):
    """
    Sugiere mejoras basadas en los datos del evento proporcionados.
    
    Args:
        event_data (dict): Datos del evento
        
    Returns:
        list: Lista de sugerencias de mejora
    """
    
    suggestions = []
    
    # Validar duración basada en tema
    if "workshop" in event_data.get("topic", "").lower():
        suggestions.append("💡 Considerar formato de workshop (2-4 horas) para mayor interactividad")
    
    # Sugerir networking
    suggestions.append("🤝 Incluir sesión de networking pre/post evento")
    
    # Sugerir grabación
    suggestions.append("📹 Considerar grabación para alcance posterior")
    
    # Sugerir materiales
    suggestions.append("📚 Preparar materiales descargables para asistentes")
    
    return suggestions


# MODIFICADO: Configuración adicional para diferentes tipos de LLM

def get_llm_specific_params(llm_type):
    """
    Retorna parámetros específicos según el tipo de LLM para optimizar propuestas.
    
    Args:
        llm_type (str): Tipo de LLM (openai, huggingface, etc.)
        
    Returns:
        dict: Parámetros optimizados
    """
    
    params = {
        "openai": {
            "temperature": 0.7,
            "max_tokens": 1500,
            "top_p": 0.9
        },
        "huggingface": {
            "temperature": 0.8,
            "max_new_tokens": 1024,
            "do_sample": True
        },
        "anthropic": {
            "temperature": 0.7,
            "max_tokens_to_sample": 1500
        }
    }
    
    return params.get(llm_type, params["openai"])


# MODIFICADO: Función principal mejorada

def create_enhanced_qa_chain(llm, llm_type="openai"):
    """
    Crea una cadena QA mejorada con parámetros optimizados para el tipo de LLM.
    
    Args:
        llm: Instancia del LLM
        llm_type (str): Tipo de LLM para optimización
        
    Returns:
        RetrievalQA: Cadena QA optimizada
    """
    
    # Obtener parámetros específicos del LLM
    llm_params = get_llm_specific_params(llm_type)
    
    # Aplicar parámetros si el LLM los soporta
    for param, value in llm_params.items():
        if hasattr(llm, param):
            setattr(llm, param, value)
    
    return get_qa_chain(llm)