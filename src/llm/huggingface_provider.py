# src/llm/huggingface_provider.py

import inspect
import os
from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFaceEndpoint
from llm.llm_provider import LLMProvider

# --- MODIFICACIÓN: Se actualiza la clase para aceptar argumentos adicionales ---
class SafeHuggingFaceEndpoint(HuggingFaceEndpoint):
    """
    Wrapper de seguridad para HuggingFaceEndpoint que maneja correctamente
    argumentos adicionales pasados por las cadenas de LangChain.
    """
    def _call(self, prompt: str, *args, **kwargs) -> str:
        # Se añaden *args para aceptar cualquier argumento posicional extra.
        # Se sigue eliminando 'stop' de los kwargs para evitar el conflicto original.
        kwargs.pop('stop', None)
        return super()._call(prompt, *args, **kwargs)

    def _stream(self, prompt: str, *args, **kwargs):
        # Se añaden *args aquí también para solucionar el error actual.
        kwargs.pop('stop', None)
        yield from super()._stream(prompt, *args, **kwargs)

# --- FIN DE MODIFICACIÓN ---


class HuggingFaceProvider(LLMProvider):
  
    def __init__(self, provider, model, params):
        super().__init__(provider, model, params)
        pass

    def _tgi_llm_instance(self, callback) -> LLM:
        print(f"[{inspect.stack()[0][3]}] Creating Hugging Face Endpoint LLM instance")

        inference_server_url = self._get_llm_url("")
        
        params: dict = {
            "endpoint_url": inference_server_url,
            "temperature": 0.7,
            "top_k": 10,
            "top_p": 0.95,
            "repetition_penalty": 1.03,
            "max_new_tokens": 1024,
            "streaming": True,
            "callbacks": [callback],
            "timeout": 120,
        }
        
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        if hf_token:
            params["huggingfacehub_api_token"] = hf_token
        
        if self.model_config and hasattr(self.model_config, 'params') and self.model_config.params:
            for param_name, param_value in self.model_config.params.items():
                if param_name in params:
                    params[param_name] = param_value
                    print(f"   Aplicando parámetro personalizado: {param_name} = {param_value}")

        # Se sigue usando nuestra clase segura, que ahora es más robusta.
        self._llm_instance = SafeHuggingFaceEndpoint(**params)

        print(f"[{inspect.stack()[0][3]}] Hugging Face Endpoint LLM instance {self._llm_instance}")
        print(f"Params: {params}")
        return self._llm_instance

    def get_llm(self, callback) -> LLM:
        return self._tgi_llm_instance(callback)