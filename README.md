# 📁 Estructura Final Completa del Proyecto

```
evento-speaker-assistant/
├── .env                                    # Variables de entorno
├── .gitignore                             # Reglas de Git ignore
├── Dockerfile                             # Imagen Docker producción
├── Dockerfile.dev                         # Imagen Docker desarrollo
├── docker-compose.yml                     # Orquestación principal
├── docker-compose.dev.yml                 # Override desarrollo
├── requirements.txt                       # Dependencias Python
├── requirements-dev.txt                   # Dependencias desarrollo
├── config.yaml                           # Configuración LLM providers
├── redis_schema.yaml                     # Schema Redis vectorstore
├── README.md                             # Documentación principal
├── Makefile                              # Comandos automatizados
├── pytest.ini                           # Configuración tests
├── setup.sh                             # Script setup automático
│
├── src/                                  # Código fuente principal
│   ├── __init__.py                       # Init módulo principal
│   ├── app.py                           # Aplicación Gradio principal (MODIFICADO)
│   │
│   ├── llm/                             # Módulos LLM
│   │   ├── __init__.py                  # Init módulo LLM
│   │   ├── client.py                    # Cliente HTTP para LLMs
│   │   ├── llm_factory.py               # Factory para instancias LLM
│   │   ├── llm_provider.py              # Clase base providers
│   │   ├── huggingface_provider.py      # Provider Hugging Face
│   │   ├── huggingface_text_gen_inference.py # TGI Hugging Face
│   │   ├── nemo_provider.py             # Provider NVIDIA NeMo
│   │   ├── openai_provider.py           # Provider OpenAI
│   │   ├── openshift_ai_vllm.py         # Provider OpenShift AI
│   │   ├── localai.py                   # Provider LocalAI
│   │   └── query_helper.py              # Helper queries eventos (MODIFICADO)
│   │
│   ├── scheduler/                       # Balanceador de carga
│   │   ├── __init__.py                  # Init scheduler
│   │   └── round_robin.py               # Algoritmo round robin
│   │
│   ├── utils/                           # Utilidades
│   │   ├── __init__.py                  # Init utils
│   │   ├── callback.py                  # Callbacks streaming
│   │   ├── config.py                    # Clases configuración
│   │   └── config_loader.py             # Cargador configuración
│   │
│   └── vector_db/                       # Bases datos vectoriales
│       ├── __init__.py                  # Init vector_db
│       ├── db_provider.py               # Clase base DB
│       ├── db_provider_factory.py       # Factory DB providers
│       ├── faiss_provider.py            # Provider FAISS
│       ├── redis_provider.py            # Provider Redis
│       ├── pgvector_provider.py         # Provider PostgreSQL+pgvector
│       └── elastic_provider.py          # Provider Elasticsearch
│
├── assets/                              # Archivos estáticos
│   ├── robot-head.ico                   # Favicon aplicación
│   ├── robot-head.svg                   # Logo SVG
│   └── event-proposals/                 # PDFs generados
│       └── .gitkeep                     # Mantener directorio
│
├── data/                                # Datos para RAG
│   ├── README.md                        # Doc datos
│   ├── speakers/                        # Información speakers
│   │   └── speakers.json                # DB speakers ejemplo
│   ├── events/                          # Datos eventos
│   │   └── events.json                  # DB eventos ejemplo
│   ├── venues/                          # Información venues
│   │   └── venues.json                  # DB venues ejemplo
│   └── topics/                          # Temas disponibles
│       └── topics.json                  # DB temas ejemplo
│
├── scripts/                             # Scripts utilidad
│   ├── init_db.py                       # Inicializar DBs vectoriales
│   ├── load_data.py                     # Cargar datos ejemplo
│   └── backup.sh                        # Script backup
│
├── tests/                               # Tests unitarios
│   ├── __init__.py                      # Init tests
│   ├── conftest.py                      # Configuración global tests
│   ├── test_app.py                      # Tests aplicación principal
│   ├── test_llm/                        # Tests módulos LLM
│   │   ├── __init__.py
│   │   ├── test_llm_factory.py          # Tests factory LLM
│   │   └── test_query_helper.py         # Tests helper queries
│   ├── test_utils/                      # Tests utilidades
│   │   ├── __init__.py
│   │   ├── test_config.py               # Tests configuración
│   │   └── test_config_loader.py        # Tests cargador config
│   └── test_vector_db/                  # Tests bases datos
│       ├── __init__.py
│       ├── test_db_factory.py           # Tests factory DB
│       └── test_providers.py            # Tests providers DB
│
├── logs/                                # Logs aplicación
│   └── .gitkeep                         # Mantener directorio
│
└── backups/                             # Backups automáticos
    └── .gitkeep                         # Mantener directorio
```

## 🔧 Archivos Principales Modificados para Eventos

### 1. **src/app.py** - Aplicación Principal
- ✅ Cambio de campos: Customer/Product → Evento/Speaker/Fecha/Horario/Ubicación/Tema
- ✅ Validaciones específicas para eventos
- ✅ Generación de PDFs con naming para eventos
- ✅ UI mejorada para gestión de eventos

### 2. **src/llm/query_helper.py** - Generador de Propuestas
- ✅ Prompt template especializado en eventos y speakers
- ✅ Estructura de propuesta profesional con 8 secciones
- ✅ Contexto específico para eventos, speakers y logística
- ✅ Generación en markdown para mejor formato

### 3. **config.yaml** - Configuración LLM
- ✅ Providers configurados (OpenAI, Hugging Face, etc.)
- ✅ Modelos optimizados para generación de texto
- ✅ Parámetros ajustados para propuestas profesionales

## 📊 Datos de Ejemplo Incluidos

### Speakers (data/speakers/speakers.json)
- Dr. Ana García - Especialista en IA
- Carlos Rodríguez - Arquitecto de Software
- María López - CTO y Emprendedora

### Eventos (data/events/events.json)
- Tech Summit 2025 - Madrid
- Developer Conference - Barcelona

### Venues (data/venues/venues.json)
- Centro de Convenciones Madrid
- Especificaciones técnicas completas

### Temas (data/topics/topics.json)
- Inteligencia Artificial
- Cloud Computing
- Desarrollo de Software

## 🚀 Quick Start

```bash
# 1. Clonar estructura
mkdir evento-speaker-assistant
cd evento-speaker-assistant

# 2. Ejecutar setup automático
wget -O setup.sh [url-del-script]
chmod +x setup.sh
./setup.sh

# 3. Configurar API key
nano .env
# Reemplazar: OPENAI_API_KEY=your_openai_api_key_here

# 4. Ejecutar aplicación
docker-compose up

# 5. Acceder a la app
# http://localhost:7860
```

## 🎯 Funcionalidades Implementadas

### ✅ Generación de Propuestas
- Formulario específico para eventos
- Validaciones de campos obligatorios
- Generación con múltiples LLM providers
- Exportación a PDF profesional

### ✅ Integración RAG
- Bases de datos vectoriales (FAISS, Redis, PostgreSQL, Elasticsearch)
- Contexto enriquecido con datos de speakers y eventos
- Embeddings para mejor relevancia

### ✅ Monitoreo y Métricas
- Prometheus metrics
- Contadores de uso por modelo
- Tiempo de respuesta
- Sistema de calificaciones

### ✅ Desarrollo y Producción
- Docker multi-stage
- Hot reload para desarrollo
- Tests automatizados
- Scripts de backup
- Makefiles para automatización

## 🔄 Flujo de Uso

1. **Usuario completa formulario**: Evento, Speaker, Fecha, Horario, Ubicación, Tema
2. **Sistema valida datos**: Campos obligatorios y formato
3. **LLM genera propuesta**: Usando RAG con contexto de la base de datos
4. **Formato profesional**: 8 secciones especializadas para eventos
5. **Exportación PDF**: Descarga inmediata de la propuesta
6. **Métricas**: Registro de uso y calificación del usuario

¡Tu **Evento Speaker Assistant** está completamente configurado y listo para generar propuestas profesionales de eventos con IA! 🎉