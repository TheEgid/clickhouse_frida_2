import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Tuple
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from frida import FridaEmbedding, FridaLangChainAdapter
from llama_index.llms.openrouter import OpenRouter
from contextlib import asynccontextmanager
from clickhouse_client import ch
from routes.health_info import router as health_info_router

# -------------------------------
# LOGGING
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

# -------------------------------
# TABLE SETUP
# -------------------------------
EMBEDDING_DIM = 1536

def create_table_if_not_exists():
    """Проверяет и создает таблицу для RAG при запуске."""
    if not ch:
        logger.error("ClickHouse client not initialized. Cannot create table.")
        return

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS rag_docs
    (
        text String,
        embedding Array(Float32),
        CONSTRAINT embedding_dim CHECK length(embedding) = {EMBEDDING_DIM}
    )
    ENGINE = MergeTree
    ORDER BY text
    """
    try:
        ch.command(create_table_query)
        logger.info("✅ Таблица 'rag_docs' проверена/создана.")

        # Опционально: добавить векторный индекс (требует ClickHouse 23.8+)
        # По умолчанию будет использоваться полный перебор (brute-force),
        # что медленно, но работает на любой версии.

        # alter_index_query = f"""
        # ALTER TABLE rag_docs
        # ADD VECTOR INDEX v1 embedding TYPE HNSWFLAT(
        #     'metric_type=Cosine', 'dim={EMBEDDING_DIM}'
        # )
        # """
        # try:
        #     ch.command(alter_index_query)
        #     logger.info("✅ Векторный индекс 'v1' проверен/создан.")
        # except Exception as e:
        #     if "INDEX_ALREADY_EXISTS" in str(e):
        #         logger.info("Векторный индекс 'v1' уже существует.")
        #     else:
        #         logger.warning(f"Не удалось создать векторный индекс (возможно, старая версия ClickHouse): {e}")

    except Exception as e:
        logger.error(f"❌ Ошибка при создании таблицы 'rag_docs': {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код при запуске
    create_table_if_not_exists()
    yield
    # Код при завершении
    if ch:
        ch.close()
        logger.info("ClickHouse connection closed.")

# -------------------------------
# INIT APP
# -------------------------------
app = FastAPI(
    title="FRIDA + ClickHouse RAG API (Manual)",
    lifespan=lifespan
)

app.include_router(health_info_router)

# -------------------------------
# SCHEMAS
# -------------------------------
class AddDoc(BaseModel):
    text: str

class Query(BaseModel):
    question: str

# -------------------------------
# MODELS SETUP
# -------------------------------
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

# -------------------------------
# ADD DOCUMENTS
# -------------------------------
@app.post("/add")
def add_document(doc: AddDoc):
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY не настроен")

        if not ch:
            raise HTTPException(status_code=500, detail="ClickHouse client not initialized.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(doc.text)

        _, embedder = setup_models(api_key)

        logger.info(f"Создание {len(chunks)} эмбеддингов...")
        embeddings = embedder.embed_documents(chunks)

        rows = [
            (chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]

        logger.info(f"Загрузка {len(rows)} строк в ClickHouse...")
        # Используем глобальный клиент 'ch'
        ch.insert(
            "rag_docs",
            rows,
            column_names=["text", "embedding"]
        )

        logger.info(f"✅ Добавлено {len(chunks)} чанков в ClickHouse")
        return {"ok": True, "chunks": len(chunks), "model": "frida"}

    except Exception as e:
        logger.exception("❌ Ошибка при добавлении документа")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# QUERY (RAG)
# -------------------------------
@app.post("/query")
def query_document(q: Query):
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY не настроен")

        if not ch:
            raise HTTPException(status_code=500, detail="ClickHouse client not initialized.")

        llm, embedder = setup_models(api_key)

        # 1. Создаем эмбеддинг для запроса
        logger.info("Создание эмбеддинга для запроса...")
        query_embedding = embedder.embed_query(q.question)

        # 2. Выполняем ручной поиск схожести
        k = 5 # Количество извлекаемых документов

        # Используем CosineDistance.
        # ORDER BY ... ASC (меньше = лучше/ближе)
        search_query = f"""
        SELECT
            text,
            CosineDistance(embedding, %(query_vec)s) AS distance
        FROM
            rag_docs
        ORDER BY
            distance ASC
        LIMIT {k}
        """

        logger.info("Выполнение векторного поиска в ClickHouse...")
        results = ch.query(
            search_query,
            parameters={"query_vec": query_embedding}
        )

        if not results.result_rows:
            logger.warning("По запросу не найдено документов.")
            return {"answer": "По вашему вопросу не найдено информации.", "sources": [], "model_used": "frida"}

        # 3. Формируем контекст и ответ
        # results.result_rows - это список кортежей [('текст1', 0.123), ('текст2', 0.456)]
        retrieved_docs_text = [row[0] for row in results.result_rows]

        context = "\n\n".join(retrieved_docs_text)
        prompt = f"Ответь на вопрос, используя контекст:\n{context}\n\nВопрос: {q.question}"

        logger.info("Генерация ответа LLM...")
        response = llm.complete(prompt)
        answer = response.text

        return {
            "answer": answer,
            "sources": [text[:200] for text in retrieved_docs_text], # Показываем первые 200 символов источников
            "model_used": "frida"
        }

    except Exception as e:
        logger.exception("❌ Ошибка при выполнении запроса")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# HEALTH CHECK
# -------------------------------
# @app.get("/health")
# def health_check():
#     if not ch:
#         raise HTTPException(status_code=500, detail="ClickHouse client not initialized.")
#     try:
#         ch.ping()
#         return {"status": "healthy", "clickhouse": "connected", "embedding_model": "frida"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"ClickHouse error: {e}")

# # -------------------------------
# # INFO ENDPOINT
# # -------------------------------
# @app.get("/info")
# def get_info():
#     if not ch:
#         raise HTTPException(status_code=500, detail="ClickHouse client not initialized.")
#     try:
#         result = ch.query("SELECT count(*) as count FROM rag_docs")
#         count = result.result_rows[0][0] if result.result_rows else 0
#         return {"documents_count": count, "table": "rag_docs", "embedding_model": "frida"}
#     except Exception as e:
#         if "Table" in str(e) and "doesn't exist" in str(e):
#             logger.warning("Таблица 'rag_docs' еще не существует.")
#             return {"documents_count": 0, "table": "rag_docs", "embedding_model": "frida", "status": "table_not_found"}
#         logger.error(f"Ошибка при получении информации: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Uvicorn entry point
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Uvicorn server stopped.")
