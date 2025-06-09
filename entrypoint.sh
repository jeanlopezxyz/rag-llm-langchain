#!/bin/bash

# Establecer variables de entorno para evitar warnings
export HF_HOME=${HF_HOME:-/app/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=true

# Crear directorios necesarios
mkdir -p $HF_HOME
mkdir -p /app/logs
mkdir -p /app/assets/event-proposals

# Ejecutar la aplicación
exec python /app/src/app.py "$@"