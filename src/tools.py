import os
from typing import Tuple

from fastapi import HTTPException
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


def get_api_key_dependency() -> str:
    """
    Зависимость FastAPI для получения и проверки наличия OPENROUTER_API_KEY.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY не настроен в переменных окружения (ошибка конфигурации сервера).",
        )
    return api_key
