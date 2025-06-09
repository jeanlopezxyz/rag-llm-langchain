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
# CONFIGURACIÓN BÁSICA
# ==========================================
load_dotenv()

if os.getenv("PYTHONHTTPSVERIFY", "1") == "0":
    os.environ["REQUESTS_CA_BUNDLE"] = ""

APP_TITLE = os.getenv("APP_TITLE", "Asistente de Eventos y Conferencias")
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def validate_configuration():
    """Valida y crea directorios necesarios."""
    required_dirs = ["assets", "data", "logs"]
    for dir_name in required_dirs:
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

# Prometheus
print(f"📊 Iniciando servidor de métricas en puerto {PROMETHEUS_PORT}...")
start_http_server(PROMETHEUS_PORT)

CHAT_COUNTER = Counter("chat_messages_total", "Total chat messages", ["model_id"])
RESPONSE_TIME = Gauge("response_time_seconds", "Response time", ["model_id"])
USER_SATISFACTION = Counter("user_satisfaction", "User satisfaction ratings", ["rating", "model_id"])

# ==========================================
# LÓGICA DE NEGOCIO (SIN CAMBIOS)
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
    """Función principal del chatbot RAG."""
    if not message.strip():
        return "Por favor, escribe una pregunta sobre los eventos."
    
    provider_id, model_id = get_provider_model(provider_model)
    if not provider_id or not model_id:
        return "❌ Error: Modelo no válido seleccionado en el Panel de Control."
    
    try:
        que = Queue()
        callback = QueueCallback(que)
        llm = llm_factory.get_llm(provider_id, model_id, callback)
        if not llm:
            return "❌ Error: No se pudo inicializar el modelo LLM."
        
        CHAT_COUNTER.labels(model_id=model_id).inc()
        qa_chain = QueryHelper.get_qa_chain(llm)
        
        result = qa_chain.invoke({"query": message})
        response = result.get("result", "No se encontró una respuesta.")
        
        if DEBUG and "source_documents" in result:
            print("\n--- INICIO DE CONTEXTO RECUPERADO (DEBUG) ---")
            if result["source_documents"]:
                for i, doc in enumerate(result["source_documents"]):
                    print(f"📄 Documento {i+1}: {doc.page_content[:200]}... | Metadata: {doc.metadata}")
            else:
                print("⚠️ No se recuperaron documentos de la base de datos vectorial.")
            print("--- FIN DE CONTEXTO RECUPERADO ---\n")

        if "source_documents" in result and result["source_documents"]:
            sources = list(set(doc.metadata.get('source', 'Desconocida') for doc in result["source_documents"]))
            if sources:
                response += "\n\n**📚 Fuentes consultadas:**\n" + "\n".join([f"• {s}" for s in sources[:3]])
        
        return response
    except Exception as e:
        print(f"❌ Error en la consulta RAG: {e}")
        return "Lo siento, no he podido procesar tu solicitud en este momento. Por favor, intenta reformular tu pregunta o vuelve a intentarlo en unos minutos."

def rate_response(rating, provider_model):
    """Función para calificar la respuesta del chat."""
    if rating and provider_model:
        provider_id, model_id = get_provider_model(provider_model)
        if model_id:
            USER_SATISFACTION.labels(rating=rating, model_id=model_id).inc()
            return f"✅ Gracias por tu calificación de {len(rating)} estrellas."
    return "Selecciona una calificación."

# ==========================================
# INTERFAZ GRADIO OPTIMIZADA
# ==========================================

# MODIFICACIÓN: Se ajusta el tema para usar un azul primario más definido.
with gr.Blocks(title=APP_TITLE, theme=gr.themes.Default(font=gr.themes.GoogleFont("Lato"), primary_hue=gr.themes.colors.blue), css=".gradio-container {max-width: 95% !important;}") as demo:
    
    gr.HTML(f"<h1 style='text-align: center; margin-bottom: 1rem;'>{APP_TITLE}</h1>")
    gr.Markdown("Plataforma de consulta para información sobre ponentes, horarios y temáticas de eventos.")

    with gr.Row(variant="panel"):
        # --- COLUMNA IZQUIERDA: PANEL DE CONTROL ---
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("### Panel de Control")
            
            with gr.Accordion("⚙️ Configuración del Modelo", open=True):
                provider_model_list = config_loader.get_provider_model_list()
                providers_dropdown = gr.Dropdown(
                    label="🤖 Modelo LLM a Utilizar",
                    choices=provider_model_list,
                    value=provider_model_list[0] if provider_model_list else None,
                    interactive=True
                )
            
            with gr.Accordion("ℹ️ Guía de Uso", open=False):
                gr.Markdown(
                    """
                    Este asistente puede responder preguntas sobre:
                    - **Horarios de charlas y eventos**
                    - **Información de ponentes**
                    - **Temas y contenidos de las sesiones**
                    - **Agendas y programación**
                    - **Ubicaciones y salas**
                    """
                )

            with gr.Accordion("⭐ Calificar Respuesta", open=False):
                rating_radio = gr.Radio(
                    ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                    label="Valora la calidad de la última respuesta",
                )
                rating_output = gr.Textbox(label="Estado", interactive=False, lines=1)
        
        # --- COLUMNA DERECHA: CHAT PRINCIPAL ---
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat_with_events,
                additional_inputs=[providers_dropdown],
                title="Consola de Consultas",
                description="Escriba su pregunta en el cuadro de abajo y presione Enter."
            )

    # --- Lógica de los Componentes ---
    rating_radio.change(
        fn=rate_response,
        inputs=[rating_radio, providers_dropdown],
        outputs=rating_output
    )

# --- INICIO DE LA APLICACIÓN ---
if __name__ == "__main__":
    print(f"🚀 Iniciando {APP_TITLE}...")
    print(f"📊 Métricas disponibles en: http://localhost:{PROMETHEUS_PORT}")
    print(f"💬 Chat disponible en: http://localhost:7860")
    
    if DEBUG:
        print("🔧 Modo DEBUG activado")
    
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path="./assets/robot-head.ico",
        show_error=True
    )