# src/llm/huggingface_provider.py

from typing import Optional, Tuple
import inspect
import os
from queue import Queue
from langchain.llms.base import LLM
# 1. Cambia la importación. Ya no necesitas tu clase local.
from langchain_huggingface import HuggingFaceEndpoint
from llm.llm_provider import LLMProvider

class HuggingFaceProvider(LLMProvider):
  
    def __init__(self, provider, model, params):
        super().__init__(provider, model, params)
        pass

    def _tgi_llm_instance(self, callback) -> LLM:
        print(f"[{inspect.stack()[0][3]}] Creating Hugging Face Endpoint LLM instance")

        inference_server_url = self._get_llm_url("")
        
        # 2. Adapta los parámetros a la nueva clase HuggingFaceEndpoint
        params: dict = {
            "endpoint_url": inference_server_url,
            "temperature": 0.7,
            "top_k": 10,
            "top_p": 0.95,
            "repetition_penalty": 1.03,
            "max_new_tokens": 1024,
            "streaming": True,
            "callbacks": [callback]
        }
        
        if self.model_config and hasattr(self.model_config, 'params') and self.model_config.params:
            for param_name, param_value in self.model_config.params.items():
                if param_name in params:
                    params[param_name] = param_value
                    print(f"   Aplicando parámetro personalizado: {param_name} = {param_value}")

        # 3. Usa la nueva clase
        self._llm_instance = HuggingFaceEndpoint(**params)

        print(f"[{inspect.stack()[0][3]}] Hugging Face Endpoint LLM instance {self._llm_instance}")
        print(f"Params: {params}")
        return self._llm_instance

    def get_llm(self, callback) -> LLM:
        return self._tgi_llm_instance(callback)