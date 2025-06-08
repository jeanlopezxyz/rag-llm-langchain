from typing import Optional, Tuple
import inspect
from langchain.llms.base import LLM
from llm.llm_provider import LLMProvider
from queue import Queue
import os

class OpenAIProvider(LLMProvider):
  def __init__(self, provider, model, params):
    super().__init__(provider, model, params)
    pass

  def _openai_llm_instance(self, callback) -> LLM:
    print(f"[{inspect.stack()[0][3]}] Creating OpenAI LLM instance")
    try:
        #from langchain.llms import OpenAI
        from langchain_openai import ChatOpenAI
    except Exception as e:
        print(
            "Missing openai libraries. Openai provider will be unavailable."
        )
        raise e

    params: dict = {
        "base_url": self._get_llm_url("https://api.openai.com/v1"),
        "openai_api_key": self._get_llm_credentials(),
        "model": self.model,
#          "model_kwargs": {},  # TODO: add model args
        "organization": None,
        "timeout": None,
        "cache": None,
        "streaming": True,
        "temperature": 0.01,
        # "top_p": 0.95,
        "verbose": False,
        "callbacks": [callback]
    }
    os.environ["OPENAI_API_KEY"] =  self._get_llm_credentials()
      # if self.model_config.params:
      #   params.update(self.model_config.params)  # override parameters

    self._llm_instance = ChatOpenAI(**params)

    print(f"[{inspect.stack()[0][3]}] OpenAI LLM instance {self._llm_instance}")
    return self._llm_instance

  def get_llm(self, callback) -> LLM:
    return self._openai_llm_instance(callback)
