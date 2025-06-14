# src/app.py
import os
import traceback
from dotenv import load_dotenv
import gradio as gr
from typing import Optional
import time
import logging

# Configuración del logger para la app principal
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from prometheus_client import Gauge, start_http_server, Counter
from utils import config_loader
import llm.query_helper as QueryHelper
from llm.llm_factory import LLMFactory

# Variables globales
llm_factory: Optional[LLMFactory] = None
knowledge_base: Optional[QueryHelper.EventKnowledgeBase] = None
CHAT_COUNTER: Optional[Counter] = None
# ... (otras variables globales)

# --- REEMPLAZA ESTA FUNCIÓN ---
def get_provider_model(provider_model_str: str) -> tuple[str, str]:
    """
    Parsea de forma robusta el string del proveedor/modelo desde el dropdown.
    """
    if not provider_model_str:
        return "", ""
    
    # Divide el string solo por el primer ':' que encuentre
    parts = provider_model_str.split(":", 1)
    
    if len(parts) == 2:
        # Usa .strip() para eliminar cualquier espacio en blanco al principio o al final
        provider = parts[0].strip()
        model = parts[1].strip()
        return provider, model
    else:
        # Si no se puede dividir, devuelve valores vacíos para que falle de forma controlada
        return "", ""
# --- FIN DEL REEMPLAZO ---

def chat_with_events(message: str, history: list, provider_model: str):
    """Función principal del chat con manejo de errores y logging robusto."""
    logging.info(f"Recibida nueva pregunta. Modelo seleccionado: '{provider_model}'. Mensaje: '{message}'")
    if not message or not message.strip():
        yield history
        return

    history.append([message, ""])
    
    try:
        start_time = time.time()
        provider_id, model_id = get_provider_model(provider_model)
        if not provider_id or not model_id:
            raise ValueError("Por favor, selecciona un proveedor y modelo desde el dropdown.")
        
        logging.debug("Obteniendo instancia de LLM desde la fábrica...")
        llm = llm_factory.get_llm(provider_id, model_id)
        
        if knowledge_base is None:
             raise ValueError("La Base de Conocimiento no fue inicializada. Revisa los logs de inicio.")
        
        logging.debug("Creando la cadena de QA para esta solicitud...")
        qa_chain = QueryHelper.create_qa_chain(knowledge_base, llm)
        
        response = ""
        logging.debug("Iniciando streaming de la respuesta...")
        for chunk in qa_chain.stream({"input": message}):
            response += chunk
            history[-1][1] = response
            yield history
        logging.info(f"Respuesta generada exitosamente. Longitud: {len(response)} caracteres.")

    except Exception as e:
        # --- CAPTURA DE ERROR MEJORADA ---
        # Imprime el traceback completo en la consola para depuración
        logging.error("Ha ocurrido una excepción durante el procesamiento del chat:", exc_info=True)
        # Prepara un mensaje de error claro para el usuario
        error_type = type(e).__name__
        error_message = f"❌ Ocurrió un error inesperado.\n\n**Tipo de Error:**\n{error_type}\n\n**Detalle:**\n{str(e)}"
        history[-1][1] = error_message
        yield history
    finally:
        logging.debug("El procesamiento de la pregunta ha finalizado.")


# ... (El resto de tu app.py, como rate_response y el bloque if __name__ == "__main__", permanece igual)
def rate_response(rating: str, provider_model: str):
    if rating and provider_model:
        _, model_id = get_provider_model(provider_model)
        if model_id and USER_SATISFACTION:
            USER_SATISFACTION.labels(rating=rating, model_id=model_id).inc()
            return "✅ ¡Gracias por tu calificación!"
    return "Para calificar, por favor selecciona las estrellas."

if __name__ == "__main__":
    try:
        load_dotenv()
        # Activar modo DEBUG si está en las variables de entorno
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            logging.getLogger().setLevel(logging.DEBUG)
            print(" MODO DEPURACIÓN ACTIVADO ".center(50, "="))

        APP_TITLE = os.getenv("APP_TITLE", "🎤 Asistente de Eventos - KCD Guatemala 2025")
        APP_SUBTITLE = os.getenv("APP_SUBTITLE", "Consulta sobre charlas, ponentes, horarios y más")
        
        # ... El resto de la inicialización ...
        start_http_server(int(os.getenv("PROMETHEUS_PORT", 8000)))
        CHAT_COUNTER = Counter("chat_messages_total", "Total de mensajes", ["model_id"])
        RESPONSE_TIME = Gauge("tiempo_de_respuesta_segundos", "Tiempo de respuesta", ["model_id"])
        USER_SATISFACTION = Counter("satisfaccion_usuario", "Calificación de satisfacción", ["rating", "model_id"])
        
        config_loader.init_config()
        llm_factory = LLMFactory()
        llm_factory.init_providers(config_loader.config)
        provider_model_list = llm_factory.get_providers()

        print("🧠 [main] Creando la Base de Conocimiento...")
        knowledge_base = QueryHelper.initialize_knowledge_base()
        print("✅ Base de Conocimiento lista.")

        with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
            gr.HTML(f"<div style='text-align: center; font-size: 2.2rem; font-weight: 700;'>{APP_TITLE}</div>")
            gr.Markdown(f"<h3 style='text-align: center; color: #4A4A4A;'>{APP_SUBTITLE}</h3>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("📂 Configuración del Asistente", open=True):
                        providers_dropdown = gr.Dropdown(
                            label="🤖 Proveedor/Modelo",
                            choices=provider_model_list,
                            value=provider_model_list[0] if provider_model_list else "",
                        )
                    with gr.Accordion("📋 Ejemplos de Preguntas", open=True):
                        EXAMPLE_QUESTIONS = [
                            "¿Cuántas charlas son en total?",
                            "¿Cuándo es el evento?",
                            "¿Cuál es la dirección del evento?",
                            "¿Qué charlas hay sobre seguridad?",
                            "¿Qué charla va a dar Sergio Méndez?",
                            "¿Ya empezó el evento?",
                        ]
                        example_buttons = [gr.Button(q, elem_classes="example-button") for q in EXAMPLE_QUESTIONS]
                    with gr.Accordion("📝 Calificación de Respuesta", open=False):
                        rating_radio = gr.Radio(
                            ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                            label="¿Cómo calificarías la respuesta?",
                        )
                        rating_output = gr.Textbox(label="Estado", interactive=False, lines=1)

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Chat", height=500, bubble_full_width=False)
                    with gr.Row():
                        textbox = gr.Textbox(show_label=False, placeholder="Escribe tu pregunta aquí...", container=False, scale=7)
                        submit_btn = gr.Button("Enviar", variant="primary", scale=1, min_width=0)

            chat_inputs = [textbox, chatbot, providers_dropdown]
            textbox.submit(fn=chat_with_events, inputs=chat_inputs, outputs=chatbot).then(lambda: "", outputs=[textbox])
            submit_btn.click(fn=chat_with_events, inputs=chat_inputs, outputs=chatbot).then(lambda: "", outputs=[textbox])
            for btn in example_buttons:
                btn.click(fn=lambda q=btn.value: q, inputs=[], outputs=[textbox]).then(fn=chat_with_events, inputs=chat_inputs, outputs=chatbot,).then(lambda: "", outputs=[textbox])
            rating_radio.change(fn=rate_response, inputs=[rating_radio, providers_dropdown], outputs=rating_output)

        print(f"🚀 [main] Iniciando la aplicación. Visita http://0.0.0.0:7860")
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True)

    except Exception as e:
        logging.error("❌❌❌ Ocurrió un error fatal durante el inicio de la aplicación.", exc_info=True)
        # Opcional: salir si la inicialización falla
        exit(1)