import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Tuple
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.clickhouse import Clickhouse
from frida import FridaEmbedding, FridaLangChainAdapter
from llama_index.llms.openrouter import OpenRouter
import clickhouse_connect

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
# INIT APP
# -------------------------------
app = FastAPI(title="FRIDA + ClickHouse RAG API")

# -------------------------------
# CLICKHOUSE CONFIG
# -------------------------------
CH_CONFIG = {
    "host": os.getenv("CLICKHOUSE_HOST", "192.168.1.77"),
    "port": int(os.getenv("CLICKHOUSE_PORT", 8123)),
    "username": os.getenv("CLICKHOUSE_USER", "default"),
    "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
    "database": os.getenv("CLICKHOUSE_DB", "default"),
}

ch = None
try:
    ch = clickhouse_connect.get_client(**CH_CONFIG)
    ch.ping()
    logger.info("✅ ClickHouse подключен успешно")
except Exception as e:
    logger.error(f"❌ Ошибка подключения к ClickHouse: {e}")

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

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(doc.text)

        _, embedder = setup_models(api_key)

        # безопасное добавление
        Clickhouse.from_texts(
            texts=chunks,
            embedding=embedder,
            table="rag_docs",
            **CH_CONFIG
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

        llm, embedder = setup_models(api_key)

        vs = Clickhouse(
            embedding=embedder,
            table="rag_docs",
            **CH_CONFIG
        )

        docs = vs.similarity_search(q.question, k=5)
        if not docs:
            return {"answer": "По вашему вопросу не найдено информации.", "sources": [], "model_used": "frida"}

        context = "\n\n".join(d.page_content for d in docs)
        prompt = f"Ответь на вопрос, используя контекст:\n{context}\n\nВопрос: {q.question}"
        response = llm.complete(prompt)
        answer = response.text

        return {
            "answer": answer,
            "sources": [d.page_content[:200] for d in docs],
            "model_used": "frida"
        }

    except Exception as e:
        logger.exception("❌ Ошибка при выполнении запроса")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/health")
def health_check():
    if not ch:
        raise HTTPException(status_code=500, detail="ClickHouse client not initialized.")
    try:
        ch.ping()
        return {"status": "healthy", "clickhouse": "connected", "embedding_model": "frida"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ClickHouse error: {e}")

# -------------------------------
# INFO ENDPOINT
# -------------------------------
@app.get("/info")
def get_info():
    if not ch:
        raise HTTPException(status_code=500, detail="ClickHouse client not initialized.")
    try:
        result = ch.query("SELECT count(*) as count FROM rag_docs")
        count = result.result_rows[0][0] if result.result_rows else 0
        return {"documents_count": count, "table": "rag_docs", "embedding_model": "frida"}
    except Exception as e:
        if "Table" in str(e) and "doesn't exist" in str(e):
            logger.warning("Таблица 'rag_docs' еще не существует.")
            return {"documents_count": 0, "table": "rag_docs", "embedding_model": "frida", "status": "table_not_found"}
        logger.error(f"Ошибка при получении информации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Uvicorn entry point
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Uvicorn server stopped.")
