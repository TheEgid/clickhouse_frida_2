from fastapi import APIRouter, HTTPException
from clickhouse_client import ch
import logging
import os
from pydantic import BaseModel
from llama_index.core.node_parser import SentenceSplitter
from tools import setup_models

logger = logging.getLogger(__name__)

router = APIRouter()

class AddDoc(BaseModel):
    text: str

class Query(BaseModel):
    question: str


@router.post("/add")
def add_document(doc: AddDoc):
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY не настроен")

        if not ch:
            raise HTTPException(status_code=500, detail="ClickHouse client not initialized.")

        splitter = SentenceSplitter(chunk_size=500, chunk_overlap=20)
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
@router.post("/query")
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
