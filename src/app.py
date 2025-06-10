#!/usr/bin/env python3
import os

# Establecer variables de entorno ANTES de cualquier import
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/app/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import time
from queue import Queue
import gradio as gr
from prometheus_client import Gauge, start_http_server, Counter
from dotenv import load_dotenv

from utils import config_loader
import llm.query_helper as QueryHelper
from utils.callback import QueueCallback

# ==========================================
# CONFIGURACIÓN PARAMETRIZADA
# ==========================================
load_dotenv()

if os.getenv("PYTHONHTTPSVERIFY", "1") == "0":
    os.environ["REQUESTS_CA_BUNDLE"] = ""

# Variables parametrizadas desde .env
APP_TITLE = os.getenv("APP_TITLE", "KCD Antigua Guatemala 2025")
EVENT_NAME = os.getenv("EVENT_NAME", "KCD Antigua Guatemala 2025")
EVENT_DATE = os.getenv("EVENT_DATE", "14 de junio de 2025")
EVENT_LOCATION = os.getenv("EVENT_LOCATION", "Centro de Convenciones Antigua, Guatemala")
EVENT_TIME = os.getenv("EVENT_TIME", "09:00 - 17:00")
EVENT_DESCRIPTION = os.getenv("EVENT_DESCRIPTION", "Asistente especializado para consultas sobre speakers, horarios y contenido del evento")
ORGANIZATION = os.getenv("ORGANIZATION", "Cloud Native Community Guatemala")

PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def validate_configuration():
    """Valida y crea directorios necesarios."""
    required_dirs = ["assets", "data", "logs"]
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("✅ Configuración validada")

validate_configuration()

# Inicialización
print(f"🔧 Inicializando {EVENT_NAME}...")
config_loader.init_config()
from llm.llm_factory import LLMFactory
llm_factory = LLMFactory()
llm_factory.init_providers(config_loader.config)

# Prometheus
print(f"📊 Métricas en puerto {PROMETHEUS_PORT}")
start_http_server(PROMETHEUS_PORT)

CHAT_COUNTER = Counter("chat_messages_total", "Total chat messages", ["model_id"])
RESPONSE_TIME = Gauge("response_time_seconds", "Response time", ["model_id"])
USER_SATISFACTION = Counter("user_satisfaction", "User satisfaction ratings", ["rating", "model_id"])

# ==========================================
# LÓGICA SIMPLIFICADA
# ==========================================

def get_provider_model(provider_model):
    """Parsea el string provider:model."""
    if provider_model is None: return "", ""
    try:
        s = provider_model.split(": ")
        return s[0], s[1] if len(s) > 1 else ""
    except:
        return "", ""

def chat_with_events(message, history, provider_model):
    """Función principal del chatbot."""
    if not message.strip():
        return f"Pregunta sobre el {EVENT_NAME}."
    
    # Respuesta directa PRIMERO (sin LLM)
    direct_answer = QueryHelper.get_direct_answer(message)
    if direct_answer:
        print(f"✅ Respuesta directa para: '{message}'")
        return direct_answer
    
    provider_id, model_id = get_provider_model(provider_model)
    if not provider_id or not model_id:
        return "❌ Error: Selecciona un modelo válido."
    
    try:
        start_time = time.time()
        
        que = Queue()
        callback = QueueCallback(que)
        llm = llm_factory.get_llm(provider_id, model_id, callback)
        if not llm:
            return "❌ Error: No se pudo inicializar el modelo."
        
        CHAT_COUNTER.labels(model_id=model_id).inc()
        
        print(f"🤖 Procesando con {model_id}...")
        qa_chain = QueryHelper.get_qa_chain(llm)
        
        result = qa_chain.invoke({"query": message})
        response = result.get("result", "No encontré información específica.")
        
        # MEJORA: Formatear respuesta con estructura
        if "source_documents" in result and result["source_documents"]:
            # Intentar formato estructurado para sesiones
            try:
                formatted_sessions = QueryHelper.format_session_response(result["source_documents"], message)
                if formatted_sessions:
                    # Si hay sesiones formateadas, usarlas
                    if len(formatted_sessions) == 1:
                        response = f"📋 **INFORMACIÓN DE LA SESIÓN:**\n\n{formatted_sessions[0]}"
                    else:
                        response = f"📋 **SESIONES ENCONTRADAS:**\n\n" + "\n\n---\n\n".join(formatted_sessions)
                    
                    # Solo añadir respuesta del LLM si no hay información estructurada suficiente
                    llm_response = result.get('result', '').strip()
                    if len(response) < 150 and llm_response and len(llm_response) > 20:
                        response += f"\n\n💬 **Información adicional:** {llm_response}"
                else:
                    # Si no se pudo formatear, usar respuesta del LLM
                    response = result.get('result', 'No encontré información específica.')
            except Exception as e:
                print(f"⚠️ Error en formato estructurado: {e}")
                # Usar respuesta original del LLM
                response = result.get('result', 'No encontré información específica.')
        else:
            # Sin documentos, usar respuesta del LLM
            response = result.get('result', 'No encontré información específica.')
        
        # Medir tiempo
        response_time = time.time() - start_time
        RESPONSE_TIME.labels(model_id=model_id).set(response_time)
        print(f"⏱️ Respuesta en {response_time:.2f}s")
        
        # Log mínimo en DEBUG
        if DEBUG and "source_documents" in result:
            print(f"📄 {len(result['source_documents'])} documentos usados")
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Respuesta de fallback parametrizada
        return f"""❌ Error procesando la consulta.

**Información básica del {EVENT_NAME}:**
📅 **Fecha:** {EVENT_DATE}
📍 **Ubicación:** {EVENT_LOCATION}  
⏰ **Horario:** {EVENT_TIME}

Intenta reformular tu pregunta."""

def rate_response(rating, provider_model):
    """Función para calificar la respuesta."""
    if rating and provider_model:
        provider_id, model_id = get_provider_model(provider_model)
        if model_id:
            USER_SATISFACTION.labels(rating=rating, model_id=model_id).inc()
            return f"✅ Gracias por tu calificación."
    return "Selecciona una calificación."

# ==========================================
# INTERFAZ LIMPIA Y PARAMETRIZADA
# ==========================================

# CSS personalizado para mejor apariencia
custom_css = """
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    text-align: center;
}

.event-info {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.config-panel {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1rem;
}

.chat-container {
    border: 1px solid #e9ecef;
    border-radius: 8px;
    background: white;
}

.examples-container {
    background: #f8f9fa;
    border-radius: 6px;
    padding: 0.75rem;
    margin-top: 0.5rem;
}
"""

with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue),
    css=custom_css
) as demo:
    
    # Header principal sin duplicación
    with gr.Row(elem_classes="main-header"):
        gr.HTML(f"""
        <div style="text-align: center;">
            <h1 style="margin: 0; font-size: 2rem;">🎯 {APP_TITLE}</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">{EVENT_DESCRIPTION}</p>
        </div>
        """)

    with gr.Row():
        # ==========================================
        # COLUMNA IZQUIERDA: CONFIGURACIÓN
        # ==========================================
        with gr.Column(scale=1, min_width=320, elem_classes="config-panel"):
            
            # Configuración del modelo
            gr.Markdown("### ⚙️ Configuración")
            
            provider_model_list = config_loader.get_provider_model_list()
            providers_dropdown = gr.Dropdown(
                label="🤖 Modelo LLM",
                choices=provider_model_list,
                value=provider_model_list[0] if provider_model_list else None,
                interactive=True
            )
            
            # Información del evento parametrizada
            gr.Markdown("### 📅 Información del Evento")
            with gr.Column(elem_classes="event-info"):
                gr.Markdown(f"""
                **{EVENT_NAME}**
                
                📅 **Fecha:** {EVENT_DATE}
                📍 **Ubicación:** {EVENT_LOCATION}
                ⏰ **Horario:** {EVENT_TIME}
                🏢 **Organiza:** {ORGANIZATION}
                """)
            
            # Guía de uso
            with gr.Accordion("💡 Guía de Uso", open=False):
                gr.Markdown(f"""
                **Puedes preguntar sobre:**
                
                📍 **Ubicación del evento**
                📅 **Fechas y horarios**
                👥 **Speakers y ponentes**
                🎯 **Contenido de las charlas**
                📋 **Crear agendas personalizadas**
                
                **Ejemplos:**
                - "¿Dónde será el {EVENT_NAME}?"
                - "¿Qué speakers presentan sobre Kubernetes?"
                - "Crea una agenda de DevOps sin conflictos"
                """)
            
            # Sistema de calificación
            with gr.Accordion("⭐ Calificar Respuesta", open=False):
                rating_radio = gr.Radio(
                    ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                    label="Califica la calidad de la respuesta",
                )
                rating_output = gr.Textbox(
                    label="Estado", 
                    interactive=False, 
                    lines=1,
                    show_label=False
                )
        
        # ==========================================
        # COLUMNA DERECHA: CHAT
        # ==========================================
        with gr.Column(scale=2, elem_classes="chat-container"):
            
            # Chat interface sin título duplicado
            chatbot = gr.ChatInterface(
                fn=chat_with_events,
                additional_inputs=[providers_dropdown],
                description=f"Escribe tu pregunta sobre el {EVENT_NAME}"
            )
            
            # Ejemplos en acordeón separado
            with gr.Accordion("💡 Ejemplos de Consultas", open=False, elem_classes="examples-container"):
                gr.Markdown(f"""
                **Prueba estas preguntas:**
                
                🏢 `¿Dónde será el {EVENT_NAME}?`
                
                📅 `¿Cuándo es el evento?`
                
                👥 `¿Qué speakers presentan sobre Kubernetes?`
                
                📋 `Crea una agenda personalizada de DevOps`
                
                🔒 `¿Hay charlas sobre seguridad?`
                
                ⏰ `¿Cuál es el horario del evento?`
                """)

    # ==========================================
    # FOOTER CON INFORMACIÓN ADICIONAL
    # ==========================================
    with gr.Row():
        gr.Markdown(f"""
        <div style="text-align: center; padding: 1rem; color: #6c757d; font-size: 0.9rem; border-top: 1px solid #e9ecef; margin-top: 1rem;">
            💡 <strong>{EVENT_NAME}</strong> • {EVENT_DATE} • {EVENT_LOCATION}<br>
            Organizado por {ORGANIZATION} • Sistema basado en IA para consultas del evento
        </div>
        """)

    # Lógica de componentes
    rating_radio.change(
        fn=rate_response,
        inputs=[rating_radio, providers_dropdown],
        outputs=rating_output
    )

# ==========================================
# INICIO OPTIMIZADO
# ==========================================
if __name__ == "__main__":
    print(f"🚀 Iniciando {APP_TITLE}...")
    print(f"📊 Métricas: http://localhost:{PROMETHEUS_PORT}")
    print(f"💬 Chat: http://localhost:7860")
    
    if DEBUG:
        print("🔧 DEBUG activado")
    
    # Estadísticas
    print(f"📋 Evento: {EVENT_NAME}")
    print(f"📅 Fecha: {EVENT_DATE}")
    print(f"📍 Ubicación: {EVENT_LOCATION}")
    print(f"🤖 Modelos disponibles: {len(config_loader.get_provider_model_list())}")
    
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path="./assets/robot-head.ico",
        show_error=True
    )