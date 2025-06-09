# import json
# import requests
# from aiohttp import ClientSession, ClientTimeout
# from typing import Dict, Optional, List, AsyncIterator, Iterator

# class Client:
#     """Cliente HTTP simple para text-generation-inference"""

#     def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None, 
#                  cookies: Optional[Dict[str, str]] = None, timeout: int = 120):
#         self.base_url = base_url
#         self.headers = headers or {}
#         self.cookies = cookies or {}
#         self.timeout = timeout

#     def generate(self, prompt: str, **kwargs):
#         """Genera texto usando la API"""
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": kwargs.get("max_new_tokens", 512),
#                 "temperature": kwargs.get("temperature", 0.7),
#                 "top_p": kwargs.get("top_p", 0.95),
#                 "do_sample": kwargs.get("do_sample", True),
#                 "return_full_text": kwargs.get("return_full_text", False)
#             }
#         }
        
#         try:
#             response = requests.post(
#                 f"{self.base_url}/generate",
#                 json=payload,
#                 headers=self.headers,
#                 cookies=self.cookies,
#                 timeout=self.timeout,
#                 verify=False
#             )
#             response.raise_for_status()
            
#             result = response.json()
            
#             # Simular estructura esperada
#             class GenerateResponse:
#                 def __init__(self, generated_text):
#                     self.generated_text = generated_text
            
#             if isinstance(result, list) and len(result) > 0:
#                 return GenerateResponse(result[0].get("generated_text", ""))
#             elif isinstance(result, dict):
#                 return GenerateResponse(result.get("generated_text", ""))
#             else:
#                 return GenerateResponse("Error en la respuesta del servidor")
                
#         except Exception as e:
#             print(f"Error en Client.generate: {e}")
#             # Retornar respuesta por defecto
#             class ErrorResponse:
#                 def __init__(self):
#                     self.generated_text = "Error: No se pudo conectar con el modelo LLM local"
#             return ErrorResponse()

#     def generate_stream(self, prompt: str, **kwargs):
#         """Genera texto con streaming"""
#         payload = {
#             "inputs": prompt,
#             "stream": True,
#             "parameters": {
#                 "max_new_tokens": kwargs.get("max_new_tokens", 512),
#                 "temperature": kwargs.get("temperature", 0.7),
#                 "top_p": kwargs.get("top_p", 0.95),
#                 "do_sample": kwargs.get("do_sample", True)
#             }
#         }
        
#         try:
#             response = requests.post(
#                 f"{self.base_url}/generate_stream",
#                 json=payload,
#                 headers=self.headers,
#                 cookies=self.cookies,
#                 timeout=self.timeout,
#                 stream=True,
#                 verify=False
#             )
            
#             for line in response.iter_lines():
#                 if line:
#                     try:
#                         data = json.loads(line.decode('utf-8'))
                        
#                         # Simular estructura de StreamResponse
#                         class StreamToken:
#                             def __init__(self, text, special=False):
#                                 self.text = text
#                                 self.special = special
                        
#                         class StreamResponse:
#                             def __init__(self, token):
#                                 self.token = token
                        
#                         if 'token' in data:
#                             token = StreamToken(data['token'].get('text', ''))
#                             yield StreamResponse(token)
                            
#                     except json.JSONDecodeError:
#                         continue
                        
#         except Exception as e:
#             print(f"Error en Client.generate_stream: {e}")
#             # Yield respuesta por defecto
#             class StreamToken:
#                 def __init__(self, text):
#                     self.text = text
#                     self.special = False
            
#             class StreamResponse:
#                 def __init__(self, token):
#                     self.token = token
            
#             yield StreamResponse(StreamToken("Error: No se pudo conectar con el modelo LLM local"))

# class AsyncClient:
#     """Cliente HTTP asíncrono simple"""

#     def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None,
#                  cookies: Optional[Dict[str, str]] = None, timeout: int = 120):
#         self.base_url = base_url
#         self.headers = headers or {}
#         self.cookies = cookies or {}
#         self.timeout = ClientTimeout(total=timeout)

#     async def generate(self, prompt: str, **kwargs):
#         """Genera texto de forma asíncrona"""
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": kwargs.get("max_new_tokens", 512),
#                 "temperature": kwargs.get("temperature", 0.7),
#                 "top_p": kwargs.get("top_p", 0.95),
#                 "do_sample": kwargs.get("do_sample", True),
#                 "return_full_text": kwargs.get("return_full_text", False)
#             }
#         }
        
#         try:
#             async with ClientSession(
#                 headers=self.headers, 
#                 cookies=self.cookies, 
#                 timeout=self.timeout
#             ) as session:
#                 async with session.post(
#                     f"{self.base_url}/generate",
#                     json=payload
#                 ) as response:
#                     result = await response.json()
                    
#                     class GenerateResponse:
#                         def __init__(self, generated_text):
#                             self.generated_text = generated_text
                    
#                     if isinstance(result, list) and len(result) > 0:
#                         return GenerateResponse(result[0].get("generated_text", ""))
#                     elif isinstance(result, dict):
#                         return GenerateResponse(result.get("generated_text", ""))
#                     else:
#                         return GenerateResponse("Error en la respuesta del servidor")
                        
#         except Exception as e:
#             print(f"Error en AsyncClient.generate: {e}")
#             class ErrorResponse:
#                 def __init__(self):
#                     self.generated_text = "Error: No se pudo conectar con el modelo LLM local"
#             return ErrorResponse()

#     async def generate_stream(self, prompt: str, **kwargs):
#         """Genera texto con streaming asíncrono"""
#         # Implementación simplificada
#         response = await self.generate(prompt, **kwargs)
        
#         class StreamToken:
#             def __init__(self, text):
#                 self.text = text
#                 self.special = False
        
#         class StreamResponse:
#             def __init__(self, token):
#                 self.token = token
        
#         # Simular streaming dividiendo la respuesta
#         words = response.generated_text.split()
#         for word in words:
#             yield StreamResponse(StreamToken(word + " "))