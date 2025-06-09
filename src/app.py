import os
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
import os
from markdown import markdown
from llm.llm_factory import LLMFactory, NVIDIA
import pdfkit
import uuid
import threading
import gradio as gr
from prometheus_client import Gauge, start_http_server, Counter
from dotenv import load_dotenv
from utils import config_loader
import llm.query_helper as QueryHelper
from scheduler.round_robin import RoundRobinScheduler
import pandas as pd
from utils.callback import QueueCallback

que = Queue()

os.environ["REQUESTS_CA_BUNDLE"] = ""
# initialization
load_dotenv()
config_loader.init_config()
llm_factory = LLMFactory()
llm_factory.init_providers(config_loader.config)

global sched

# Parameters - MODIFICADO: Cambio de título y directorio
APP_TITLE = os.getenv("APP_TITLE", "Evento Speaker Assistant 🎤")
PDF_FILE_DIR = "event-proposals"  # MODIFICADO: Era "proposal-docs"
TIMEOUT = int(os.getenv("TIMEOUT", 30))

# Start Prometheus metrics server
start_http_server(8000)

# Create metric
FEEDBACK_COUNTER = Counter(
    "feedback_stars", "Number of feedbacks by stars", ["stars", "model_id"]
)
MODEL_USAGE_COUNTER = Counter(
    "model_usage", "Number of times a model was used", ["model_id"]
)
REQUEST_TIME = Gauge(
    "request_duration_seconds", "Time spent processing a request", ["model_id"]
)


def create_scheduler():
    global sched
    provider_model_weight_list = config_loader.get_provider_model_weight_list()
    # initialize scheduler
    sched = RoundRobinScheduler(provider_model_weight_list)


create_scheduler()


# PDF Generation - MODIFICADO: Cambio de naming
def get_pdf_file(session_id):
    return os.path.join("./assets", PDF_FILE_DIR, f"evento-{session_id}.pdf")  # MODIFICADO


def create_pdf(text, session_id):
    try:
        # Crear directorio si no existe
        output_dir = os.path.join("./assets", PDF_FILE_DIR)
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = get_pdf_file(session_id)
        html_text = markdown(text, output_format="html4")
        
        # Configurar opciones de wkhtmltopdf para mejor formato
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        pdfkit.from_string(html_text, output_filename, options=options)
    except Exception as e:
        print(f"Error creando PDF: {e}")


# Function to initialize all star ratings to 0
def initialize_feedback_counters(model_id):
    for star in range(1, 6):  # For star ratings 1 to 5
        FEEDBACK_COUNTER.labels(stars=str(star), model_id=model_id).inc(0)


def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata["source"] not in unique_list:
            unique_list.append(item.metadata["source"])
    return unique_list


lock = threading.Lock()


def stream(llm, que, input_text, session_id, model_id) -> Generator:
    # Create a Queue
    job_done = object()

    qa_chain = QueryHelper.get_qa_chain(llm)

    # Create a function to call - this will run in a thread
    def task():
        MODEL_USAGE_COUNTER.labels(model_id=model_id).inc()
        # Call this function at the start of your application
        initialize_feedback_counters(model_id)
        with lock:
            start_time = (
                time.perf_counter()
            )  # start and end time to get the precise timing of the request
            try:
                resp = qa_chain.invoke({"query": input_text})
                end_time = time.perf_counter()
                sources = remove_source_duplicates(resp["source_documents"])
                REQUEST_TIME.labels(model_id=model_id).set(end_time - start_time)
                create_pdf(resp["result"], session_id)
                if len(sources) != 0:
                    que.put("\n\n**📚 Fuentes consultadas:**\n")
                    for source in sources:
                        que.put(f"• {str(source)}\n")
            except Exception as e:
                print(f"Error en generación: {e}")
                que.put("❌ Error ejecutando la solicitud. Contacta al administrador.")

            que.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = que.get(True, timeout=100)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue


# MODIFICADO: Nueva función para generar query de eventos
def ask_llm(provider_model, event_name, speaker_name, date, time_slot, location, topic, additional_info=""):
    que = Queue()
    callback = QueueCallback(que)
    session_id = str(uuid.uuid4())
    provider_id, model_id = get_provider_model(provider_model)
    llm = llm_factory.get_llm(provider_id, model_id, callback)

    # MODIFICADO: Query especializada para eventos de charlas
    query = f"""Generar una propuesta profesional completa para evento de charla con speaker que incluya:

INFORMACIÓN DEL EVENTO:
- Nombre del evento: '{event_name}'
- Speaker: '{speaker_name}'
- Fecha: '{date}'
- Horario: '{time_slot}'
- Ubicación: '{location}'
- Tema principal: '{topic}'
- Información adicional: '{additional_info}'

SOLICITUD:
Crear una propuesta integral que incluya: biografía del speaker, agenda detallada de la sesión, objetivos de aprendizaje, requerimientos técnicos, expectativas de audiencia, logística y coordinación, promoción y marketing, y actividades de seguimiento.

La propuesta debe ser profesional, detallada y específica para este evento de charla."""

    print(f"Generando propuesta para evento: {event_name}")

    for next_token, content in stream(llm, que, query, session_id, model_id):
        # Generate the download link HTML
        download_link_html = f'<input type="hidden" id="pdf_file" name="pdf_file" value="/file={get_pdf_file(session_id)}" />'
        yield content, download_link_html


def get_provider_model(provider_model):
    if provider_model is None:
        return "", ""
    s = provider_model.split(": ")
    return s[0], s[1]


def is_provider_visible():
    return config_loader.config.type == "all"


def get_selected_provider():
    if config_loader.config.type == "round_robin":
        return sched.get_next()

    provider_list = config_loader.get_provider_model_weight_list()
    if len(provider_list) > 0:
        return provider_list[0]

    return None


# Gradio implementation - MODIFICADO: CSS mejorado para eventos
css = """
#output-container {
    font-size: 0.9rem !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.event-form {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.width_200 {
     width: 200px;
}

.width_300 {
     width: 300px;
}

.width_100 {
     width: 100px;
}

.width_50 {
     width: 50px;
}

.add_provider_bu {
    max-width: 200px;
}

.generate-btn {
    background: linear-gradient(45deg, #28a745, #20c997) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
}

.clear-btn {
    background: linear-gradient(45deg, #dc3545, #fd7e14) !important;
    border: none !important;
    color: white !important;
}

.event-title {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1rem;
    font-size: 1.5rem;
    font-weight: bold;
}
"""


def get_provider_list_as_df():
    provider_list = config_loader.get_provider_display_list()
    df = pd.DataFrame(provider_list)
    df = df.rename(
        columns={
            "provider_name": "Provider",
            "enabled": "Enabled",
            "url": "URL",
            "model_name": "Model",
        }
    )
    return df


# MODIFICADO: Interfaz completamente rediseñada para eventos
with gr.Blocks(title=APP_TITLE, css=css) as demo:
    with gr.Tab("🎤 Generador de Propuestas de Eventos"):
        provider_model_list = config_loader.get_provider_model_list()
        provider_model_var = gr.State()
        provider_visible = is_provider_visible()
        
        # Header
        gr.HTML(f'<div class="event-title">🎤 {APP_TITLE}</div>')
        gr.Markdown("### Genera propuestas profesionales para eventos de charlas con speakers usando IA")
        
        with gr.Row():
            with gr.Column(scale=1):
                # MODIFICADO: Sección de configuración LLM
                with gr.Group():
                    gr.Markdown("#### ⚙️ Configuración del Modelo")
                    providers_dropdown = gr.Dropdown(
                        label="🤖 Provider/Modelo LLM", 
                        choices=provider_model_list,
                        value=provider_model_list[0] if provider_model_list else None,
                        info="Selecciona el modelo de IA a utilizar"
                    )
                    model_text = gr.HTML(visible=not provider_visible)
                
                # MODIFICADO: Formulario específico para eventos
                with gr.Group():
                    gr.Markdown("#### 📅 Información del Evento")
                    event_name_box = gr.Textbox(
                        label="🎯 Nombre del Evento", 
                        placeholder="Ej: TechConf 2025, AI Summit Madrid...",
                        info="Nombre oficial del evento o conferencia"
                    )
                    
                    speaker_name_box = gr.Textbox(
                        label="👤 Nombre del Speaker", 
                        placeholder="Ej: Dr. Ana García, Carlos Rodríguez...",
                        info="Nombre completo del ponente o conferenciante"
                    )
                    
                    with gr.Row():
                        date_box = gr.Textbox(
                            label="📅 Fecha", 
                            placeholder="Ej: 2025-03-15",
                            info="Fecha del evento (YYYY-MM-DD)"
                        )
                        time_box = gr.Textbox(
                            label="🕐 Horario", 
                            placeholder="Ej: 14:00 - 15:30",
                            info="Hora de inicio y fin de la charla"
                        )
                    
                    location_box = gr.Textbox(
                        label="📍 Ubicación", 
                        placeholder="Ej: Auditorio Principal, Sala 3, Centro de Convenciones...",
                        info="Lugar específico donde se realizará la charla"
                    )
                    
                    topic_box = gr.Textbox(
                        label="💡 Tema Principal", 
                        placeholder="Ej: Inteligencia Artificial en la Medicina, DevOps en la Nube...",
                        info="Tema central de la presentación"
                    )
                    
                    additional_info_box = gr.Textbox(
                        label="📝 Información Adicional (Opcional)", 
                        placeholder="Audiencia objetivo, nivel técnico, objetivos específicos...",
                        lines=3,
                        info="Detalles extra que puedan ser relevantes"
                    )
                
                # MODIFICADO: Botones de acción mejorados
                with gr.Row():
                    submit_button = gr.Button("🚀 Generar Propuesta", elem_classes="generate-btn", variant="primary")
                    clear_button = gr.Button("🗑️ Limpiar Formulario", elem_classes="clear-btn")
                
                # MODIFICADO: Sistema de calificación mejorado
                with gr.Group():
                    gr.Markdown("#### ⭐ Califica la Propuesta Generada")
                    radio = gr.Radio(
                        ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], 
                        label="Calificación",
                        info="¿Qué te pareció la propuesta generada?"
                    )
                    output_rating = gr.Textbox(
                        elem_id="source-container", 
                        interactive=False, 
                        label="Estado de Calificación",
                        visible=True
                    )

            with gr.Column(scale=2):
                # MODIFICADO: Área de output mejorada
                with gr.Group():
                    gr.Markdown("#### 📋 Propuesta Generada")
                    lines = 25 if provider_visible else 30
                    output_answer = gr.Textbox(
                        label="📄 Propuesta del Evento",
                        interactive=True,
                        lines=lines,
                        elem_id="output-container",
                        placeholder="La propuesta aparecerá aquí una vez generada...",
                        show_label=False
                    )
                    
                    with gr.Row():
                        download_button = gr.Button("📥 Descargar como PDF", variant="secondary")
                        copy_button = gr.Button("📋 Copiar al Portapapeles", variant="secondary")
                    
                    download_link_html = gr.HTML(visible=False)

        # Event handlers - MODIFICADO
        download_button.click(
            None,
            [],
            [],
            js="() => window.open(document.getElementById('pdf_file').value, '_blank')",
        )

        def update_models(selected_provider, provider_model):
            provider_id, model_id = get_provider_model(selected_provider)
            m = f"<div style='padding:10px; background:#f0f0f0; border-radius:5px;'><span id='model_id'><strong>Modelo activo:</strong> {model_id}</span></div>"
            return {provider_model_var: selected_provider, model_text: m}

        providers_dropdown.change(
            update_models,
            inputs=[providers_dropdown, provider_model_var],
            outputs=[provider_model_var, model_text],
        )

        # MODIFICADO: Validación mejorada para eventos
        def validate_generate_input(provider, event_name, speaker_name, date, time_slot, location, topic):
            errors = []
            
            if not provider:
                errors.append("🤖 Provider/Modelo")
            if not event_name or len(event_name.strip()) < 3:
                errors.append("🎯 Nombre del evento (mínimo 3 caracteres)")
            if not speaker_name or len(speaker_name.strip()) < 2:
                errors.append("👤 Nombre del speaker (mínimo 2 caracteres)")
            if not topic or len(topic.strip()) < 5:
                errors.append("💡 Tema principal (mínimo 5 caracteres)")
            
            if errors:
                error_msg = "Por favor completa los siguientes campos obligatorios:\n" + "\n".join([f"• {error}" for error in errors])
                raise gr.Error(error_msg)
            
            return True

        submit_button.click(
            validate_generate_input,
            inputs=[providers_dropdown, event_name_box, speaker_name_box, date_box, time_box, location_box, topic_box],
        ).success(
            ask_llm,
            inputs=[providers_dropdown, event_name_box, speaker_name_box, date_box, time_box, location_box, topic_box, additional_info_box],
            outputs=[output_answer, download_link_html],
        )
        
        # MODIFICADO: Limpiar formulario mejorado
        clear_button.click(
            lambda: [None, "", "", "", "", "", "", "", "", "🗑️ Formulario limpiado correctamente"],
            inputs=[],
            outputs=[
                providers_dropdown,
                event_name_box,
                speaker_name_box,
                date_box,
                time_box,
                location_box,
                topic_box,
                additional_info_box,
                output_answer,
                output_rating,
            ],
        )

        # MODIFICADO: Sistema de feedback mejorado
        @radio.input(inputs=[radio, provider_model_var], outputs=output_rating)
        def get_feedback(star_rating, provider_model):
            if not star_rating or not provider_model:
                return "Selecciona una calificación y asegúrate de tener un modelo seleccionado"
            
            # Convertir rating visual a número
            star_count = len(star_rating)
            
            provider_id, model_id = get_provider_model(provider_model)
            print(f"Feedback recibido - Modelo: {provider_model}, Rating: {star_count} estrellas")
            
            # Increment the counter based on the star rating received
            FEEDBACK_COUNTER.labels(stars=str(star_count), model_id=model_id).inc()
            
            feedback_messages = {
                1: "😞 Gracias por tu feedback. Trabajaremos para mejorar.",
                2: "😐 Entendemos que hay espacio para mejorar. ¡Gracias!",
                3: "😊 ¡Gracias! Una calificación promedio nos ayuda a crecer.",
                4: "😃 ¡Excelente! Nos alegra que te haya gustado la propuesta.",
                5: "🎉 ¡Fantástico! Una propuesta de 5 estrellas. ¡Gracias!"
            }
            
            return feedback_messages.get(star_count, "✅ ¡Gracias por tu calificación!")

    # MODIFICADO: Tab de configuración con mejor organización (mantener el código original de configuración)
    with gr.Tab("⚙️ Configuración") as provider_tab:
        # ... (mantener todo el código de configuración original, solo cambiar el título)
        with gr.Accordion("🔧 Tipo de Configuración"):
            type_dropdown = gr.Dropdown(
                ["round_robin", "all"],
                label="Tipo de Balanceador",
                value=config_loader.config.type,
                info="Selecciona cómo usar los LLM providers (round_robin: rotación, all: mostrar todos)",
            )

            update_type_btn = gr.Button("💾 Guardar Configuración", elem_classes="add_provider_bu")

            def update_type(type):
                config_loader.config.type = type
                create_scheduler()
                return {
                    type_dropdown: gr.Dropdown(
                        ["round_robin", "all"],
                        label="Tipo de Balanceador",
                        value=type,
                        info="Selecciona cómo usar los LLM providers (round_robin: rotación, all: mostrar todos)",
                    )
                }

            update_type_btn.click(
                update_type, inputs=[type_dropdown], outputs=[type_dropdown]
            ).success(None, outputs=[type_dropdown], js="window.location.reload()")

        with gr.Accordion("🤖 Gestión de Providers"):
            df = get_provider_list_as_df()
            dataframe_ui = gr.Dataframe(value=df, interactive=False, label="Providers Configurados")
            add_btn = gr.Button("➕ Agregar Nuevo Provider", elem_classes="add_provider_bu")

        # ... (resto del código de configuración original)

        def initialize(provider_model):
            if provider_model is None:
                provider_model_tuple = get_selected_provider()
                if provider_model_tuple is not None:
                    provider_model = provider_model_tuple[0]
            print(f"Inicializando con modelo: {provider_model}")
            
            provider_id, model_id = get_provider_model(provider_model) if provider_model else ("", "")
            provider_visible = is_provider_visible()
            provider_model_list = config_loader.get_provider_model_list()
            
            p_dropdown = gr.Dropdown(
                choices=provider_model_list,
                label="🤖 Provider/Modelo LLM",
                value=provider_model,
                info="Selecciona el modelo de IA a utilizar"
            )
            
            m = f"<div style='padding:10px; background:#f0f0f0; border-radius:5px;'><span id='model_id'><strong>Modelo activo:</strong> {model_id}</span></div>"
            df = get_provider_list_as_df()
            df_component = gr.Dataframe(
                value=df, interactive=False, label="Providers Configurados"
            )
            td = gr.Dropdown(
                ["round_robin", "all"],
                label="Tipo de Balanceador",
                value=config_loader.config.type,
                info="Selecciona cómo usar los LLM providers",
            )
            return {
                providers_dropdown: p_dropdown,
                provider_model_var: provider_model,
                model_text: m,
                dataframe_ui: df_component,
                type_dropdown: td,
            }

        demo.load(
            initialize,
            inputs=[provider_model_var],
            outputs=[providers_dropdown, provider_model_var, model_text, dataframe_ui, type_dropdown],
        )


if __name__ == "__main__":
    print(f"🚀 Iniciando {APP_TITLE}...")
    print(f"📊 Métricas disponibles en: http://localhost:8000")
    
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path="./assets/robot-head.ico",
        allowed_paths=["assets"],
        show_error=True
    )