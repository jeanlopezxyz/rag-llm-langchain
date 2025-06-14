# src/llm/llm_factory.py

import logging
from llm.llm_provider import LLMProvider
from llm.openshift_ai_vllm import OpenShiftAIvLLM
# Descomenta e importa los proveedores que realmente existen en tu proyecto
# from llm.huggingface_provider import HuggingFaceProvider
# from llm.nemo_provider import NeMoProvider
# from llm.openai_provider import OpenAIProvider 
from langchain.llms.base import LLM

# Configuración del logger
logger = logging.getLogger(__name__)

# Constantes para los nombres de los proveedores
OPENSHIFT_AI_VLLM = "OpenShift AI (vLLM)"
# HUGGING_FACE = "Hugging Face"
# NVIDIA = "NVIDIA"
# OPENAI = "OpenAI"

class LLMFactory:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}

    def _create_key(self, provider, model):
        return f"{provider}:{model}"
    
    def init_providers(self, config):
        """
        Inicializa solo los proveedores y modelos que están marcados como 'enabled: true'.
        """
        self._providers = {}
        logger.info("Iniciando la inicialización de proveedores en LLMFactory...")
        
        provider_map = {
            # Aquí solo necesitas los proveedores que realmente vas a usar
            OPENSHIFT_AI_VLLM: OpenShiftAIvLLM,
            # OPENAI: OpenAIProvider,
        }

        # Accedemos a la configuración de forma segura usando getattr
        providers_config_obj = getattr(config, 'llm_providers', None)
        if not providers_config_obj or not hasattr(providers_config_obj, 'providers'):
            logger.warning("No se encontró la sección 'llm_providers.providers' en la configuración.")
            return

        # Iteramos sobre el diccionario de proveedores
        for provider_key, provider_cfg in providers_config_obj.providers.items():
            # --- INICIO DE LA CORRECCIÓN CLAVE ---
            # 1. Verificamos si el PROVEEDOR está habilitado
            if not getattr(provider_cfg, 'enabled', False):
                logger.info(f"Proveedor '{provider_cfg.name}' deshabilitado en config.yaml. Omitiendo.")
                continue

            provider_name = getattr(provider_cfg, 'name', '')
            if provider_class := provider_map.get(provider_name):
                
                # Iteramos sobre los modelos del proveedor habilitado
                for model_key, model_cfg in provider_cfg.models.items():
                    # 2. Verificamos si el MODELO está habilitado
                    if not getattr(model_cfg, 'enabled', False):
                        logger.info(f"Modelo '{model_cfg.name}' del proveedor '{provider_name}' deshabilitado. Omitiendo.")
                        continue
                    
                    model_name = getattr(model_cfg, 'name', '')
                    params = getattr(model_cfg, 'params', {})
                    key = self._create_key(provider_name, model_name)
                    
                    logger.info(f"✅ Registrando proveedor/modelo habilitado: '{key}'")
                    self._providers[key] = provider_class(provider_name, model_name, params)
            # --- FIN DE LA CORRECCIÓN CLAVE ---
            else:
                logger.warning(f"Proveedor '{provider_name}' no encontrado en provider_map de la fábrica.")

    def get_llm(self, provider, model) -> LLM:
        """Obtiene una instancia de LLM del proveedor y modelo especificados."""
        key = self._create_key(provider, model)
        provider_instance = self._providers.get(key)
        if provider_instance:
            return provider_instance.get_llm()
        else:
            raise ValueError(f"Provider/Model no encontrado en la fábrica: '{key}'. Proveedores disponibles: {list(self._providers.keys())}")

    def get_providers(self) -> list:
        """Devuelve una lista de todos los proveedores y modelos disponibles."""
        return list(self._providers.keys())