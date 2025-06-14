# src/llm/openshift_ai_vllm.py

import inspect
from langchain.llms.base import LLM
from llm.llm_provider import LLMProvider
import os
import httpx

class OpenShiftAIvLLM(LLMProvider):
  def __init__(self, provider, model, params):
    super().__init__(provider, model, params)

  def _openshift_ai_vllm_instance(self) -> LLM:
    print(f"[{inspect.stack()[0][3]}] Creating OpenShift AI vLLM instance")
    try:
      from langchain_community.llms.vllm import VLLMOpenAI
    except ImportError as e:
      print("Missing vLLM libraries. VLLMOpenAI provider will be unavailable.")
      raise e
    
    creds = self._get_llm_credentials() or "dummy-api-key"
    os.environ["OPENAI_API_KEY"] = creds

    # Usamos los parámetros pasados durante la inicialización
    params_dict = {
        "model_name": self.model,
        "openai_api_base": self._get_llm_url(""),
        "temperature": self.params.get("temperature", 0.1),
        "max_tokens": self.params.get("max_tokens", 1024),
        "verbose": os.getenv("DEBUG", "false").lower() == "true",
    }
    
    async_client = httpx.AsyncClient(verify=False)
    http_client = httpx.Client(verify=False)
    
    self._llm_instance = VLLMOpenAI(**params_dict, async_client=async_client, http_client=http_client)

    print(f"[{inspect.stack()[0][3]}] OpenShift AI vLLM instance created.")
    return self._llm_instance

  # Versión final: no acepta argumentos.
  def get_llm(self) -> LLM:
    return self._openshift_ai_vllm_instance()