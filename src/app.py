# src/app.py
import os
import traceback
import re  # ← AGREGAR ESTA IMPORTACIÓN
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
knowledge_base: Optional[QueryHelper.EnhancedEventKnowledgeBase] = None  # ← CAMBIAR TIPO
CHAT_COUNTER: Optional[Counter] = None
RESPONSE_TIME: Optional[Gauge] = None
USER_SATISFACTION: Optional[Counter] = None

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

def chat_with_events(message: str, history: list, provider_model: str):
    """Función principal del chat con filtrado anti-alucinación mejorado."""
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
        
        # Frases problemáticas que indican estructura innecesaria
        forbidden_phrases = [
            "pregunta:", "respuesta:", "consulta del usuario:", "respuesta directa:",
            "se espera", "se esperan", "el evento se lleva a cabo",
            "la dirección es", "el enlace es", "https://", "www.", ".com",
            "123 ", "ejemplo", "example"
            # REMOVIDO: "presentará una charla titulada" - esta frase puede ser parte de respuestas válidas
        ]
        
        final_response = ""
        
        for chunk in qa_chain.stream({"input": message}):
            response += chunk
            
            # Filtrar en tiempo real SOLO frases muy problemáticas
            clean_chunk = chunk
            chunk_lower = chunk.lower()
            
            # Detectar SOLO frases que realmente indican estructura problemática
            contains_forbidden = any(phrase in chunk_lower for phrase in [
                "pregunta:", "respuesta:", "consulta del usuario:", "respuesta directa:"
            ])
            
            if not contains_forbidden:
                final_response += clean_chunk
                history[-1][1] = final_response
                yield history
            else:
                # Si detectamos estructura problemática, parar y limpiar
                logging.warning(f"Detectada estructura problemática en: {chunk}")
                break
        
        # POST-PROCESAMIENTO: Limpiar respuesta final
        if final_response:
            clean_response = final_response.strip()
            
            # Remover estructuras de pregunta/respuesta (solo al final, no durante streaming)
            patterns_to_remove = [
                r'^Pregunta:.*?Respuesta:\s*',
                r'^CONSULTA DEL USUARIO:.*?RESPUESTA DIRECTA:\s*',
                r'Pregunta:.*?Respuesta:\s*',
            ]
            
            for pattern in patterns_to_remove:
                clean_response = re.sub(pattern, '', clean_response, flags=re.IGNORECASE | re.DOTALL)
            
            # Limpiar líneas vacías múltiples
            clean_response = re.sub(r'\n\s*\n', '\n', clean_response).strip()
            
            # NO cortar la respuesta si es larga - dejar que el LLM complete la lista
            # Solo verificar que la respuesta final no esté vacía
            if clean_response and len(clean_response.strip()) > 5:
                history[-1][1] = clean_response
            else:
                history[-1][1] = "No disponible."
                
            yield history
        else:
            history[-1][1] = "No disponible."
            yield history
        
        logging.info(f"Respuesta procesada exitosamente. Longitud final: {len(history[-1][1])} caracteres.")

    except Exception as e:
        logging.error("Ha ocurrido una excepción durante el procesamiento del chat:", exc_info=True)
        error_message = f"❌ Error: {str(e)}"
        history[-1][1] = error_message
        yield history
    finally:
        logging.debug("El procesamiento de la pregunta ha finalizado.")

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
        
        # Inicializar métricas
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

        # Interfaz moderna con Gradio 4
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
                            "¿Qué charlas hay sobre seguridad?",
                            "¿Qué charlas hay sobre IA?",
                            "¿Qué charla da Jean Paul López?",
                            "¿A qué hora habla Víctor Castellanos?",
                            "¿Cuáles son las sesiones de la tarde?",
                        ]
                        example_buttons = [gr.Button(q, elem_classes="example-button") for q in EXAMPLE_QUESTIONS]
                    with gr.Accordion("📝 Calificación de Respuesta", open=False):
                        rating_radio = gr.Radio(
                            ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                            label="¿Cómo calificarías la respuesta?",
                        )
                        rating_output = gr.Textbox(label="Estado", interactive=False, lines=1)

                with gr.Column(scale=2):
                    # Chatbot moderno sin parámetros deprecados
                    chatbot = gr.Chatbot(
                        label="Chat", 
                        height=500,
                        type="tuples"  # Usar formato de tuples en lugar de messages
                    )
                    with gr.Row():
                        textbox = gr.Textbox(
                            show_label=False, 
                            placeholder="Escribe tu pregunta aquí...", 
                            container=False, 
                            scale=7
                        )
                        submit_btn = gr.Button("Enviar", variant="primary", scale=1, min_width=0)

            # Eventos
            chat_inputs = [textbox, chatbot, providers_dropdown]
            
            textbox.submit(
                fn=chat_with_events, 
                inputs=chat_inputs, 
                outputs=chatbot
            ).then(
                lambda: "", 
                outputs=[textbox]
            )
            
            submit_btn.click(
                fn=chat_with_events, 
                inputs=chat_inputs, 
                outputs=chatbot
            ).then(
                lambda: "", 
                outputs=[textbox]
            )
            
            # Botones de ejemplo
            for btn in example_buttons:
                btn.click(
                    fn=lambda q=btn.value: q, 
                    inputs=[], 
                    outputs=[textbox]
                ).then(
                    fn=chat_with_events, 
                    inputs=chat_inputs, 
                    outputs=chatbot
                ).then(
                    lambda: "", 
                    outputs=[textbox]
                )
            
            rating_radio.change(
                fn=rate_response, 
                inputs=[rating_radio, providers_dropdown], 
                outputs=rating_output
            )

        print(f"🚀 [main] Iniciando la aplicación. Visita http://0.0.0.0:7860")
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True)

    except Exception as e:
        logging.error("❌❌❌ Ocurrió un error fatal durante el inicio de la aplicación.", exc_info=True)
        exit(1)