from typing import Optional, Tuple
import inspect
import os
from queue import Queue
from langchain.llms.base import LLM
from llm.huggingface_text_gen_inference import HuggingFaceTextGenInference
from llm.llm_provider import LLMProvider
from llm.client import Client, AsyncClient  # Importar los clientes

class HuggingFaceProvider(LLMProvider):
  
    def __init__(self, provider, model, params):
        super().__init__(provider, model, params)
        pass

    def _tgi_llm_instance(self, callback) -> LLM:
        """Note: TGI does not support specifying the model, it is an instance per model."""
        print(f"[{inspect.stack()[0][3]}] Creating Hugging Face TGI LLM instance")

        inference_server_url = self._get_llm_url("")
        
        # Crear los clientes client y async_client
        client = Client(inference_server_url, timeout=120)
        async_client = AsyncClient(inference_server_url, timeout=120)

        params: dict = {
            "inference_server_url": inference_server_url,
            "client": client,           # ← AGREGADO: Campo requerido
            "async_client": async_client, # ← AGREGADO: Campo requerido
            "cache": None,
            "temperature": 0.7,         # ← AUMENTADO de 0.01
            "top_k": 10,
            "top_p": 0.95,
            "repetition_penalty": 1.03,
            "max_new_tokens": 1024,     # ← AGREGADO explícitamente
            "streaming": True,
            "verbose": False,
            "callbacks": [callback]
        }
        
        # Si hay parámetros específicos del modelo, aplicarlos
        if self.model_config and hasattr(self.model_config, 'params') and self.model_config.params:
            for param_name, param_value in self.model_config.params.items():
                if param_name in params:
                    params[param_name] = param_value
                    print(f"   Aplicando parámetro personalizado: {param_name} = {param_value}")

        self._llm_instance = HuggingFaceTextGenInference(**params)

        print(f"[{inspect.stack()[0][3]}] Hugging Face TGI LLM instance {self._llm_instance}")
        print(f"Params: {params}")
        return self._llm_instance

    def get_llm(self, callback) -> LLM:
        return self._tgi_llm_instance(callback)