import os
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
import gradio as gr
from prometheus_client import Gauge, start_http_server, Counter
from dotenv import load_dotenv
from utils import config_loader
import llm.query_helper as QueryHelper
from scheduler.round_robin import RoundRobinScheduler
from utils.callback import QueueCallback
import uuid

# ==========================================
# CONFIGURACIÓN BÁSICA
# ==========================================
load_dotenv()

if os.getenv("PYTHONHTTPSVERIFY", "1") == "0":
    os.environ["REQUESTS_CA_BUNDLE"] = ""

APP_TITLE = os.getenv("APP_TITLE", "Evento Speaker Assistant 🎤")
TIMEOUT = int(os.getenv("TIMEOUT", 30))
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Validación inicial
def validate_configuration():
    """Valida configuración al inicio."""
    required_dirs = ["assets", "data", "logs"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
    print("✅ Validación de configuración completada")

validate_configuration()

# Inicialización
print("🔧 Inicializando configuración...")
config_loader.init_config()
print("🤖 Inicializando factory de LLMs...")
from llm.llm_factory import LLMFactory
llm_factory = LLMFactory()
llm_factory.init_providers(config_loader.config)

global sched

# Prometheus
print(f"📊 Iniciando servidor de métricas en puerto {PROMETHEUS_PORT}...")
start_http_server(PROMETHEUS_PORT)

CHAT_COUNTER = Counter("chat_messages_total", "Total chat messages", ["model_id"])
RESPONSE_TIME = Gauge("response_time_seconds", "Response time", ["model_id"])
USER_SATISFACTION = Counter("user_satisfaction", "User satisfaction ratings", ["rating", "model_id"])

def create_scheduler():
    """Crea el scheduler de forma segura."""
    global sched
    try:
        provider_model_weight_list = config_loader.get_provider_model_weight_list()
        if not provider_model_weight_list:
            print("⚠️ No hay providers habilitados.")
            return
        sched = RoundRobinScheduler(provider_model_weight_list)
        print("✅ Scheduler inicializado correctamente")
    except Exception as e:
        print(f"❌ Error inicializando scheduler: {e}")

create_scheduler()

# ==========================================
# FUNCIONES DE CHAT RAG
# ==========================================

def get_provider_model(provider_model):
    """Parsea el string provider:model."""
    if provider_model is None:
        return "", ""
    try:
        s = provider_model.split(": ")
        return s[0], s[1] if len(s) > 1 else ""
    except:
        return "", ""

def chat_with_events(message, history, provider_model):
    """Función principal del chatbot RAG."""
    if not message.strip():
        return "Por favor, escribe una pregunta sobre los eventos."
    
    try:
        provider_id, model_id = get_provider_model(provider_model)
        if not provider_id or not model_id:
            return "❌ Error: Modelo no válido seleccionado"
        
        # Crear callback para streaming
        que = Queue()
        callback = QueueCallback(que)
        
        # Obtener LLM
        llm = llm_factory.get_llm(provider_id, model_id, callback)
        if not llm:
            return "❌ Error: No se pudo inicializar el modelo LLM"
        
        # Incrementar contador
        CHAT_COUNTER.labels(model_id=model_id).inc()
        
        if DEBUG:
            print(f"🗣️ Pregunta: {message}")
            print(f"🤖 Modelo: {model_id}")
        
        # Crear cadena QA
        start_time = time.perf_counter()
        qa_chain = QueryHelper.get_qa_chain(llm)
        
        # Procesar pregunta
        try:
            # Ejecutar consulta RAG
            result = qa_chain.invoke({"query": message})
            response = result["result"]
            
            # Agregar fuentes si existen
            if "source_documents" in result and result["source_documents"]:
                sources = []
                for doc in result["source_documents"]:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source = doc.metadata['source']
                        if source not in sources:
                            sources.append(source)
                
                if sources:
                    response += "\n\n**📚 Fuentes consultadas:**\n"
                    for source in sources[:3]:  # Máximo 3 fuentes
                        response += f"• {source}\n"
            
            end_time = time.perf_counter()
            RESPONSE_TIME.labels(model_id=model_id).set(end_time - start_time)
            
            if DEBUG:
                print(f"✅ Respuesta generada en {end_time - start_time:.2f}s")
            
            return response
            
        except Exception as e:
            print(f"❌ Error en consulta RAG: {e}")
            return f"❌ Error procesando tu pregunta: {str(e)}"
            
    except Exception as e:
        print(f"❌ Error en chat_with_events: {e}")
        return f"❌ Error: {str(e)}"

# ==========================================
# EJEMPLOS DE PREGUNTAS
# ==========================================

EXAMPLE_QUESTIONS = [
    "¿Qué charlas se dan hoy?",
    "¿Cuáles son las charlas relacionadas con Inteligencia Artificial?",
    "Generame una agenda de las charlas de Machine Learning que no se crucen en horario",
    "¿Cuál es la charla del Dr. García?",
    "¿Cuántas charlas son en total?",
    "¿Dónde queda el local del evento?",
    "¿Cuántas charlas dará el speaker María López?",
    "¿A qué hora es la charla de DevOps?",
    "¿Qué charlas hay disponibles el viernes?",
    "Muéstrame el horario completo del evento"
]

def rate_response(rating, provider_model):
    """Función para calificar respuesta."""
    if rating and provider_model:
        provider_id, model_id = get_provider_model(provider_model)
        if model_id:
            USER_SATISFACTION.labels(rating=str(rating), model_id=model_id).inc()
            return f"✅ Gracias por tu calificación: {rating} estrellas"
    return "Selecciona una calificación"

# ==========================================
# INTERFAZ GRADIO
# ==========================================

# CSS personalizado
css = """
#chat-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.example-questions {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.chat-title {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1rem;
    font-size: 1.5rem;
    font-weight: bold;
}

.provider-config {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #dee2e6;
}
"""

# Interfaz principal
with gr.Blocks(title=APP_TITLE, css=css) as demo:
    
    # Header
    gr.HTML(f'<div class="chat-title">🎤 {APP_TITLE} - Chatbot</div>')
    gr.Markdown("### Pregúntame sobre eventos, charlas, speakers, horarios y ubicaciones")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Configuración del modelo
            with gr.Group():
                gr.Markdown("#### ⚙️ Configuración del Modelo")
                provider_model_list = config_loader.get_provider_model_list()
                providers_dropdown = gr.Dropdown(
                    label="🤖 Provider/Modelo LLM",
                    choices=provider_model_list,
                    value=provider_model_list[0] if provider_model_list else None,
                    info="Modelo de IA para responder preguntas"
                )
            
            # Ejemplos de preguntas
            with gr.Group():
                gr.Markdown("#### 💡 Preguntas de Ejemplo")
                gr.Markdown("Haz clic para copiar:")
                
                for i, question in enumerate(EXAMPLE_QUESTIONS[:8]):
                    gr.Markdown(f"**{i+1}.** {question}")
            
            # Sistema de calificación
            with gr.Group():
                gr.Markdown("#### ⭐ Califica la Respuesta")
                rating_radio = gr.Radio(
                    ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                    label="¿Qué te pareció la respuesta?",
                    info="Tu feedback nos ayuda a mejorar"
                )
                rating_output = gr.Textbox(
                    label="Estado de Calificación",
                    interactive=False,
                    visible=True
                )
        
        with gr.Column(scale=2):
            # FIX: Chat interface con type="messages" para evitar warning de tuples
            chatbot = gr.ChatInterface(
                chat_with_events,
                additional_inputs=[providers_dropdown],
                type="messages"  # FIX: Usar messages en lugar de tuples
            )
    
    # Event handler para calificación
    rating_radio.change(
        rate_response,
        inputs=[rating_radio, providers_dropdown],
        outputs=rating_output
    )

    # Inicialización
    def initialize():
        provider_model_list = config_loader.get_provider_model_list()
        return gr.Dropdown(
            choices=provider_model_list,
            value=provider_model_list[0] if provider_model_list else None
        )
    
    demo.load(initialize, outputs=providers_dropdown)

if __name__ == "__main__":
    print(f"🚀 Iniciando {APP_TITLE} - Chatbot...")
    print(f"📊 Métricas disponibles en: http://localhost:{PROMETHEUS_PORT}")
    print(f"💬 Chat disponible en: http://localhost:7860")
    
    if DEBUG:
        print("🔧 Modo DEBUG activado")
    
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path="./assets/robot-head.ico",
        allowed_paths=["assets"],
        show_error=True
    )