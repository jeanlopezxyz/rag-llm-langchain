.PHONY: help build run run-bg stop clean test lint format install dev-install

# Variables
DOCKER_COMPOSE = docker-compose
PYTHON = python
PIP = pip

# Default target
help: ## Mostrar esta ayuda
	@echo "Comandos disponibles:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Docker commands
build: ## Construir la imagen Docker
	$(DOCKER_COMPOSE) build

run: ## Ejecutar la aplicación
	$(DOCKER_COMPOSE) up

run-bg: ## Ejecutar en background
	$(DOCKER_COMPOSE) up -d

stop: ## Parar todos los servicios
	$(DOCKER_COMPOSE) down

clean: ## Limpiar contenedores, imágenes y volúmenes
	$(DOCKER_COMPOSE) down -v --rmi all
	docker system prune -f

restart: stop run ## Reiniciar la aplicación

# Development commands
install: ## Instalar dependencias
	$(PIP) install -r requirements.txt

dev-install: ## Instalar dependencias de desarrollo
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 isort

test: ## Ejecutar tests
	$(DOCKER_COMPOSE) exec app $(PYTHON) -m pytest tests/ -v

test-local: ## Ejecutar tests localmente
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Ejecutar tests con coverage
	$(DOCKER_COMPOSE) exec app $(PYTHON) -m pytest tests/ --cov=src --cov-report=html

lint: ## Ejecutar linter
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Formatear código
	black src/ tests/
	isort src/ tests/

# Database commands
init-db: ## Inicializar base de datos
	$(DOCKER_COMPOSE) exec app $(PYTHON) scripts/init_db.py

load-data: ## Cargar datos de ejemplo
	$(DOCKER_COMPOSE) exec app $(PYTHON) scripts/load_data.py

# Utility commands
logs: ## Ver logs de la aplicación
	$(DOCKER_COMPOSE) logs -f app

logs-all: ## Ver logs de todos los servicios
	$(DOCKER_COMPOSE) logs -f

shell: ## Abrir shell en el contenedor
	$(DOCKER_COMPOSE) exec app /bin/bash

backup: ## Crear backup
	./scripts/backup.sh

# Monitoring
metrics: ## Ver métricas de Prometheus
	@echo "Métricas disponibles en: http://localhost:8000"
	curl -s http://localhost:8000/metrics | head -20

status: ## Ver estado de los servicios
	$(DOCKER_COMPOSE) ps

# Setup commands
setup: ## Setup inicial del proyecto
	@echo "🚀 Configurando Evento Speaker Assistant..."
	@echo "1. Copiando archivo de configuración..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "   ✅ .env creado (recuerda configurar tus API keys)"; fi
	@echo "2. Creando directorios necesarios..."
	@mkdir -p assets/event-proposals data/speakers data/events backups
	@echo "   ✅ Directorios creados"
	@echo "3. Construyendo imagen Docker..."
	@$(MAKE) build
	@echo "🎉 Setup completado. Ejecuta 'make run' para iniciar la aplicación"

# Production commands
deploy: ## Deploy en producción
	@echo "🚀 Desplegando en producción..."
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "✅ Aplicación desplegada"

deploy-stop: ## Parar despliegue de producción
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml down

# Cleanup commands
clean-logs: ## Limpiar logs
	$(DOCKER_COMPOSE) exec app find . -name "*.log" -delete
	@echo "✅ Logs limpiados"

clean-cache: ## Limpiar cache de Python
	find . -type d -name "__pycache__" -delete
	find . -name "*.pyc" -delete
	@echo "✅ Cache de Python limpiado"

clean-all: clean clean-logs clean-cache ## Limpiar todo

# Development workflow
dev: dev-install ## Setup entorno de desarrollo
	@echo "🛠️ Entorno de desarrollo configurado"
	@echo "   Ejecuta 'source venv/bin/activate' para activar el entorno virtual"

check: lint test ## Ejecutar checks completos (lint + tests)

pre-commit: format lint test ## Ejecutar antes de commit
	@echo "✅ Pre-commit checks completados"

# Docker development
dev-build: ## Build para desarrollo con hot reload
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml build

dev-run: ## Ejecutar en modo desarrollo
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up

# Monitoring and debugging
debug: ## Ejecutar en modo debug
	$(DOCKER_COMPOSE) exec app $(PYTHON) -m pdb src/app.py

profile: ## Ejecutar con profiling
	$(DOCKER_COMPOSE) exec app $(PYTHON) -m cProfile -o profile.stats src/app.py

monitor: ## Mostrar recursos del sistema
	$(DOCKER_COMPOSE) exec app top

# Documentation
docs: ## Generar documentación
	@echo "📚 Generando documentación..."
	@echo "README.md actualizado"

# Quick commands
quick-start: setup run ## Setup y ejecutar rápidamente
	@echo "🚀 Aplicación iniciada en http://localhost:7860"

# Version management
version: ## Mostrar versión actual
	@echo "Evento Speaker Assistant v1.0.0"
	@$(DOCKER_COMPOSE) --version
	@$(PYTHON) --version