# Dockerfile para producción - Evento Speaker Assistant
# Optimizado para compatibilidad con Podman/OCI

# ===========================
# Stage 1: Builder
# ===========================
FROM python:3.11-slim as builder

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Crear directorio para dependencias
RUN mkdir -p /install

# Copiar requirements.txt
COPY requirements.txt .

# --- CORRECCIÓN AQUÍ ---
# Actualizar pip a la última versión para eliminar el notice y asegurar compatibilidad.
RUN pip install --upgrade pip

# Instalar dependencias Python con el pip ya actualizado
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ===========================
# Stage 2: Production
# ===========================
FROM python:3.11-slim

# Metadata de la imagen
LABEL maintainer="Evento Speaker Assistant"
LABEL version="1.0.0"
LABEL description="Aplicación para generar propuestas de eventos con speakers usando IA"

# Instalar dependencias del sistema para runtime
RUN apt-get update && apt-get install -y \
    wkhtmltopdf \
    xvfb \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Establecer directorio de trabajo
WORKDIR /app

# Copiar dependencias Python desde builder
COPY --from=builder /install /usr/local

# Copiar código fuente
COPY src/ ./src/
COPY config.yaml .
COPY redis_schema.yaml .

# Copiar y crear directorios necesarios
COPY assets/ ./assets/
RUN mkdir -p assets/event-proposals data/speakers data/events data/venues logs

# Configurar permisos
RUN chown -R appuser:appuser /app
RUN chmod +x src/app.py

# Variables de entorno para producción
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Variables de entorno de la aplicación
ENV APP_TITLE="Evento Speaker Assistant 🎤"
ENV CONFIG_FILE=config.yaml
ENV DB_TYPE=FAISS
ENV PROMETHEUS_PORT=8000

# Cambiar a usuario no-root
USER appuser

# Exponer puertos
EXPOSE 7860
EXPOSE 8000

# Comando por defecto
CMD ["python", "src/app.py"]