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

    # --- MODIFICACIÓN CLAVE: Separar parámetros del constructor y de la API ---
    # Parámetros que el constructor de VLLMOpenAI acepta directamente
    constructor_params = {
        "model_name": self.model,
        "openai_api_base": self._get_llm_url(""),
        "verbose": os.getenv("DEBUG_MODE", "false").lower() == "true",
    }
    
    # Parámetros para la llamada a la API que van dentro de `model_kwargs`
    model_kwargs = {}

    model_params_from_config = self.params or {}
    for key, value in model_params_from_config.items():
        if key == "max_new_tokens":
            # `max_tokens` es un argumento directo del constructor
            constructor_params["max_tokens"] = value
        else:
            # El resto (temperature, top_p, frequency_penalty) va a model_kwargs
            model_kwargs[key] = value

    # Asegurar valores por defecto si no están en la config
    model_kwargs.setdefault("temperature", 0.01)
    model_kwargs.setdefault("frequency_penalty", 0.2) 

    constructor_params.setdefault("max_tokens", 512)

    print(f"Final constructor parameters for VLLMOpenAI: {constructor_params}")
    print(f"Final model_kwargs for API call: {model_kwargs}")
    # --- FIN DE LA MODIFICACIÓN ---
    
    async_client = httpx.AsyncClient(verify=False)
    http_client = httpx.Client(verify=False)
    
    self._llm_instance = VLLMOpenAI(
        **constructor_params, 
        model_kwargs=model_kwargs, # Pasar los parámetros de API aquí
        async_client=async_client, 
        http_client=http_client
    )

    print(f"[{inspect.stack()[0][3]}] OpenShift AI vLLM instance created.")
    return self._llm_instance

  def get_llm(self) -> LLM:
    return self._openshift_ai_vllm_instance()
