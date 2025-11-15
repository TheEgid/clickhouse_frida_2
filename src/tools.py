from typing import Tuple
from frida import FridaEmbedding, FridaLangChainAdapter
from llama_index.llms.openrouter import OpenRouter


def setup_models(api_key: str) -> Tuple[OpenRouter, FridaLangChainAdapter]:
    llm = OpenRouter(
        model="tngtech/deepseek-r1t2-chimera:free",
        max_tokens=3000,
        temperature=0.3,
        api_key=api_key,
        context_window=4096,
        system_prompt="Ты - полезный AI-ассистент. Всегда отвечай на русском языке.",
    )
    embedder = FridaLangChainAdapter(FridaEmbedding())
    return llm, embedder
