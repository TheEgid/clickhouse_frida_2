from fastapi import APIRouter, HTTPException, Depends
from clickhouse_client import ch
import logging
from pydantic import BaseModel
from llama_index.core.node_parser import SentenceSplitter
from tools import setup_models, get_api_key_dependency
import uuid
import json

logger = logging.getLogger(__name__)

router = APIRouter()


class AddDoc(BaseModel):
    text: str
    metadata: dict = None


class Query(BaseModel):
    question: str
    top_k: int = 5


@router.post("/add")
async def add_document(doc: AddDoc, api_key: str = Depends(get_api_key_dependency)):
    """Добавление документа с использованием новой структуры таблицы."""
    try:
        if not ch:
            raise HTTPException(
                status_code=500, detail="ClickHouse client not initialized."
            )

        splitter = SentenceSplitter(chunk_size=500, chunk_overlap=20)
        chunks = splitter.split_text(doc.text)

        _, embedder = setup_models(api_key)

        logger.info(f"Создание {len(chunks)} эмбеддингов...")
        embeddings = embedder.embed_documents(chunks)

        rows = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            rows.append(
                [
                    str(uuid.uuid4()),  # id
                    chunk,  # text
                    embedding,  # embedding
                    json.dumps(
                        {
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "source_info": doc.metadata or {}
                        }
                    ),  # metadata
                ]
            )

        ch.insert("novaya", rows, column_names=["id", "text", "embedding", "metadata"])

        logger.info(f"✅ Добавлено {len(chunks)} чанков в ClickHouse")
        return {"ok": True, "chunks": len(chunks), "model": "frida", "table": "novaya"}

    except Exception as e:
        logger.exception("❌ Ошибка при добавлении документа")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_document(q: Query, api_key: str = Depends(get_api_key_dependency)):
    """Выполнение RAG-запроса с использованием векторного индекса ClickHouse."""
    try:
        if not ch:
            raise HTTPException(
                status_code=500, detail="ClickHouse client not initialized."
            )

        llm, embedder = setup_models(api_key)

        logger.info("Создание эмбеддинга для запроса...")
        query_embedding = embedder.embed_query(q.question)

        # современный синтаксис ClickHouse для векторного поиска
        search_query = """
        SELECT
            id,
            text,
            metadata,
            cosineDistance(embedding, {query_vec:Array(Float32)}) AS distance
        FROM novaya
        ORDER BY distance ASC
        LIMIT {top_k:Int32}
        """

        logger.info("Выполнение векторного поиска в ClickHouse...")
        results = ch.query(
            search_query, parameters={"query_vec": query_embedding, "top_k": q.top_k}
        )

        if not results.result_rows:
            logger.warning("По запросу не найдено документов.")
            return {
                "answer": "По вашему вопросу не найдено информации.",
                "sources": [],
                "model_used": "frida",
            }

        # 3. Формируем контекст и ответ
        retrieved_docs = []
        for row in results.result_rows:
            doc_id, text, metadata, distance = row
            retrieved_docs.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": json.loads(metadata) if metadata else {},
                    "distance": float(distance),
                }
            )

        retrieved_docs_text = [doc["text"] for doc in retrieved_docs]

        # logger.info(retrieved_docs_text)

        context = "\n\n".join(retrieved_docs_text)
        prompt = (
            f"Ответь на вопрос, используя контекст:\n{context}\n\nВопрос: {q.question}"
        )

        # logger.info(context)

        logger.info("Генерация ответа LLM...")
        response = llm.complete(prompt)
        answer = response.text

        return {
            "answer": answer,
            "sources": [
                {
                    "id": doc["id"],
                    "preview": doc["text"][:200],
                    "metadata": doc["metadata"],
                    "distance": doc["distance"],
                }
                for doc in retrieved_docs
            ],
            "model_used": "frida",
        }

    except Exception as e:
        logger.exception("❌ Ошибка при выполнении запроса")
        raise HTTPException(status_code=500, detail=str(e))


# Дополнительный endpoint для проверки структуры таблицы
@router.get("/table-info")
async def get_table_info(api_key: str = Depends(get_api_key_dependency)):
    """Получение информации о структуре таблицы."""
    try:
        table_query = """
        DESCRIBE TABLE novaya
        """

        result = ch.query(table_query)
        columns = []
        for row in result.result_rows:
            columns.append(
                {
                    "name": row[0],
                    "type": row[1],
                    "default_type": row[2],
                    "default_expression": row[3],
                    "comment": row[4],
                    "codec_expression": row[5],
                    "ttl_expression": row[6],
                }
            )

        return {"table": "novaya", "columns": columns}

    except Exception as e:
        logger.exception("❌ Ошибка при получении информации о таблице")
        raise HTTPException(status_code=500, detail=str(e))
