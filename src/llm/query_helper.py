from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
from vector_db.db_provider_factory import FAISS, DBFactory

############################
# LLM chain implementation for AGENDAS
############################

db_factory = DBFactory()

# PROMPT ESPECIALIZADO PARA AGENDAS PERSONALIZADAS
agenda_prompt_template = """
### [INST]
Eres un **ASISTENTE EXPERTO EN AGENDAS PERSONALIZADAS** para el **KCD Antigua Guatemala 2025**.

**INFORMACIÓN DEL EVENTO:**
- **Evento**: KCD Antigua Guatemala 2025
- **Fecha**: 14 de junio de 2025  
- **Ubicación**: Centro de Convenciones Antigua, Guatemala
- **Horario**: 09:00 - 17:00
- **Salas**: Salón Landívar, Salón El Obispo, Salón Don Pedro

**TU ESPECIALIDAD:**
Cuando detectes palabras como "agenda", "crea", "planifica", "sin conflictos", "horario personalizado":

1. **IDENTIFICAR** qué tecnologías/temas le interesan al usuario
2. **SELECCIONAR** sesiones relevantes sin conflictos de horario  
3. **CREAR CRONOGRAMA** organizado y claro
4. **VERIFICAR** que no haya solapamientos temporales

**FORMATO PARA AGENDAS:**
```
🗓️ **AGENDA PERSONALIZADA - KCD Antigua Guatemala 2025**

**📋 Tu agenda:**
- X sesiones seleccionadas
- Sin conflictos de horario ✅
- Temas: [tecnologías cubiertas]

**⏰ CRONOGRAMA:**

**09:00 - 09:35** | Salón Landívar
🎯 **[Nombre sesión]**
👤 [Speaker] ([Empresa])
📚 [Track] | 🏷️ [Tecnologías]

[Repetir para cada sesión]

**💡 Notas:**
- Agenda optimizada sin conflictos
- [Observaciones adicionales]
```

**PARA CONSULTAS NORMALES** (no de agenda), responde como asistente experto del KCD.

CONTEXTO DE SESIONES:
{context}

**INSTRUCCIONES:**
- Si es solicitud de agenda: usa el formato estructurado arriba
- Si es consulta normal: responde directamente sobre el KCD
- SIEMPRE menciona horarios específicos y salas
- VERIFICA que las sesiones seleccionadas no se solapen en tiempo

PREGUNTA DEL USUARIO:
{question}

Responde como experto en el KCD Antigua Guatemala 2025.
[/INST]
"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=agenda_prompt_template, input_variables=["context", "question"]
)

def is_agenda_request(user_question: str) -> bool:
    """Detectar si es una solicitud de agenda personalizada."""
    agenda_indicators = [
        'agenda', 'cronograma', 'planifica', 'crea', 'genera', 'arma',
        'horario personalizado', 'sin conflictos', 'no se crucen',
        'evitando', 'programa', 'itinerario', 'planifica mi día'
    ]
    
    user_lower = user_question.lower()
    return any(indicator in user_lower for indicator in agenda_indicators)

def get_qa_chain(llm):
    """
    Crear cadena QA que maneja tanto consultas normales como agendas.
    """
    # Usar PGVECTOR con la nueva colección
    db_type = "PGVECTOR"
    
    try:
        print(f"🗃️ Inicializando DB para KCD con agendas: {db_type}")
        base_retriever = db_factory.get_retriever(db_type)
        print(f"✅ {db_type} configurado correctamente")
        
        # Configuración para agendas y consultas normales
        if hasattr(base_retriever, 'vectorstore'):
            retriever = base_retriever.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 8,  # Suficientes sesiones para crear agendas
                }
            )
            print(f"🎯 Retriever configurado para agendas y consultas (k=8)")
        else:
            retriever = base_retriever
            print(f"🎯 Usando retriever base para {db_type}")
        
    except Exception as e:
        print(f"⚠️ Error configurando {db_type}: {e}")
        raise ValueError(f"No se pudo configurar la base de datos vectorial: {e}")

    # WRAPPER INTELIGENTE que hereda de BaseRetriever
    class SmartKCDRetriever(BaseRetriever):
        """Retriever inteligente que maneja agendas y consultas normales."""
        
        def __init__(self, base_retriever):
            super().__init__()
            self.base_retriever = base_retriever
        
        def _get_relevant_documents(
            self, 
            query: str, 
            *, 
            run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            """Método requerido por BaseRetriever."""
            return self.get_relevant_documents(query)
        
        def get_relevant_documents(self, query: str) -> List[Document]:
            """Método principal para obtener documentos relevantes."""
            print(f"🔍 DEBUG KCD: Consultando '{query}'")
            
            is_agenda = is_agenda_request(query)
            if is_agenda:
                print("📅 Detectada SOLICITUD DE AGENDA")
                return self.handle_agenda_request(query)
            else:
                print("💬 Consulta normal sobre KCD")
                return self.handle_normal_query(query)
        
        def handle_agenda_request(self, query: str) -> List[Document]:
            """Manejar solicitudes de agenda personalizada."""
            print("🎯 Procesando solicitud de agenda...")
            
            # Extraer tecnologías/temas mencionados
            tech_keywords = [
                'kubernetes', 'k8s', 'helm', 'security', 'seguridad',
                'devops', 'gitops', 'cicd', 'ci/cd', 'automation',
                'machine learning', 'ml', 'ia', 'ai', 'chatbot'
            ]
            
            user_lower = query.lower()
            mentioned_techs = [tech for tech in tech_keywords if tech in user_lower]
            
            print(f"🏷️ Tecnologías detectadas: {mentioned_techs}")
            
            # Buscar sesiones relevantes
            docs = []
            
            if mentioned_techs:
                # Buscar por tecnologías específicas
                for tech in mentioned_techs[:3]:  # Máximo 3 tecnologías
                    try:
                        tech_docs = self.base_retriever.vectorstore.similarity_search(tech, k=4)
                        docs.extend(tech_docs)
                        print(f"   📄 {len(tech_docs)} documentos para '{tech}'")
                    except Exception as e:
                        print(f"   ⚠️ Error buscando '{tech}': {e}")
            else:
                # Búsqueda general para agendas
                general_terms = ['agenda', 'sesión', 'charla', 'KCD']
                for term in general_terms:
                    try:
                        term_docs = self.base_retriever.vectorstore.similarity_search(term, k=3)
                        docs.extend(term_docs)
                        print(f"   📄 {len(term_docs)} documentos para '{term}'")
                        if len(docs) >= 8:  # Suficientes documentos
                            break
                    except Exception as e:
                        print(f"   ⚠️ Error buscando '{term}': {e}")
            
            # Remover duplicados por session_id
            unique_docs = []
            seen_sessions = set()
            
            for doc in docs:
                session_id = doc.metadata.get('session_id')
                if session_id and session_id not in seen_sessions:
                    seen_sessions.add(session_id)
                    unique_docs.append(doc)
            
            print(f"📊 {len(unique_docs)} sesiones únicas encontradas para agenda")
            
            # Ordenar por horario si está disponible
            try:
                unique_docs.sort(key=lambda x: x.metadata.get('start_time', ''))
            except:
                pass  # Si no se puede ordenar, continuar
            
            return unique_docs[:8]  # Máximo 8 sesiones para agenda
        
        def handle_normal_query(self, query: str) -> List[Document]:
            """Manejar consultas normales sobre el KCD."""
            try:
                docs = self.base_retriever.get_relevant_documents(query)
                print(f"📄 {len(docs)} documentos encontrados para consulta normal")
                
                if docs:
                    for i, doc in enumerate(docs[:3]):  # Mostrar primeros 3
                        session_name = doc.metadata.get('session_name', 'Sin nombre')
                        speakers = doc.metadata.get('speakers_info', 'Sin speaker')
                        start_time = doc.metadata.get('start_time', 'Sin horario')
                        
                        print(f"   {i+1}. {session_name}")
                        print(f"      👤 {speakers}")
                        print(f"      ⏰ {start_time}")
                else:
                    print("⚠️ No se encontraron documentos relevantes")
                
                return docs
                
            except Exception as e:
                print(f"❌ Error en consulta normal: {e}")
                return []

    # Usar el retriever inteligente
    smart_retriever = SmartKCDRetriever(retriever)

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

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def format_kcd_query(user_question):
    """Formatear consultas para el KCD."""
    user_lower = user_question.lower()
    
    if is_agenda_request(user_question):
        return f"Solicitud de agenda personalizada para KCD Antigua Guatemala 2025: {user_question}"
    elif any(word in user_lower for word in ["speaker", "ponente", "quién"]):
        return f"Consulta sobre speakers del KCD: {user_question}"
    elif any(word in user_lower for word in ["horario", "cuándo", "hora"]):
        return f"Consulta sobre horarios del KCD: {user_question}"
    elif any(word in user_lower for word in ["dónde", "ubicación", "sala"]):
        return f"Consulta sobre ubicaciones del KCD: {user_question}"
    else:
        return f"Consulta sobre KCD Antigua Guatemala 2025: {user_question}"

def suggest_kcd_questions(user_question):
    """Sugerir preguntas relacionadas."""
    user_lower = user_question.lower()
    
    if is_agenda_request(user_question):
        return [
            "💡 'Crea agenda de Kubernetes y seguridad sin conflictos'",
            "💡 'Agenda solo de la mañana sobre DevOps'",
            "💡 'Planifica charlas de IA que no se solapen'",
            "💡 'Agenda evitando horario de almuerzo'"
        ]
    else:
        return [
            "¿Qué presenta Sergio Méndez?",
            "¿Cuándo habla Jorge Romero?", 
            "¿Hay charlas sobre Kubernetes?",
            "¿Dónde es el evento KCD?"
        ]

def enhance_query_for_kcd(user_question):
    """Mejorar consultas para el KCD."""
    formatted_query = format_kcd_query(user_question)
    suggestions = suggest_kcd_questions(user_question)
    
    user_lower = user_question.lower()
    
    return {
        "formatted_query": formatted_query,
        "original_question": user_question,
        "is_agenda_request": is_agenda_request(user_question),
        "suggestions": suggestions,
        "event_context": {
            "name": "KCD Antigua Guatemala 2025",
            "date": "14 de junio de 2025",
            "location": "Centro de Convenciones Antigua",
            "type": "Evento CNCF gratuito presencial"
        }
    }

# Exportar funciones principales
__all__ = [
    'get_qa_chain',
    'format_kcd_query',
    'enhance_query_for_kcd',
    'is_agenda_request'
]