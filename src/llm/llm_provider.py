# src/llm/llm_provider.py
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from langchain.llms.base import LLM
# La siguiente importación es para los type hints, no causa problemas de ejecución
from utils.config import ProviderConfig, ModelConfig

class LLMProvider(ABC):
    """
    Clase base abstracta para todos los proveedores de LLM.
    Define la interfaz común que deben seguir y proporciona métodos de ayuda
    para acceder a la configuración.
    """
    def __init__(self, provider: str, model: str, params: dict):
        """
        Constructor de la clase base.
        """
        self.provider = provider
        self.model = model
        self.params = params
        self._llm_instance = None

    @abstractmethod
    def get_llm(self) -> LLM:
        """
        Método abstracto que las clases hijas deben implementar.
        Debe devolver una instancia de un LLM compatible con LangChain.
        """
        pass

    # --- MODIFICACIÓN CLAVE ---
    # Se cambia la sintaxis del type hint para máxima compatibilidad con
    # diferentes versiones de Python.
    def _get_llm_config(self) -> Tuple[Optional[ProviderConfig], Optional[ModelConfig]]:
        """
        Función de ayuda para obtener la configuración del proveedor y modelo
        desde el cargador de configuración global.
        La importación se hace aquí para evitar dependencias circulares.
        """
        # Se importa localmente para evitar el ciclo de importación
        from utils import config_loader
        return config_loader.get_provider_model(self.provider, self.model)

    def _get_llm_url(self, default_url: str = "") -> str:
        """
        Obtiene la URL del endpoint para el modelo, usando la URL del proveedor
        como fallback.
        """
        provider_cfg, model_cfg = self._get_llm_config()
        
        # La URL específica del modelo tiene prioridad
        if model_cfg and hasattr(model_cfg, 'url') and model_cfg.url:
            return model_cfg.url
            
        # Si no, se usa la URL general del proveedor
        if provider_cfg and hasattr(provider_cfg, 'url') and provider_cfg.url:
            return provider_cfg.url
            
        return default_url

    # --- MODIFICACIÓN CLAVE ---
    # Se cambia también aquí para consistencia y compatibilidad.
    def _get_llm_credentials(self) -> Optional[str]:
        """
        Obtiene las credenciales para el modelo, usando las credenciales del
        proveedor como fallback.
        """
        provider_cfg, model_cfg = self._get_llm_config()

        # Las credenciales específicas del modelo tienen prioridad
        if model_cfg and hasattr(model_cfg, 'credentials') and model_cfg.credentials:
            return model_cfg.credentials
            
        # Si no, se usan las credenciales generales del proveedor
        if provider_cfg and hasattr(provider_cfg, 'credentials') and provider_cfg.credentials:
            return provider_cfg.credentials
            
        return None
