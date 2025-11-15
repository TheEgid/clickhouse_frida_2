from fastapi import APIRouter, HTTPException
from clickhouse_client import ch
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
def health_check():
    if not ch:
        raise HTTPException(status_code=500, detail="ClickHouse client not initialized.")
    try:
        ch.ping()
        return {"status": "healthy", "clickhouse": "connected", "embedding_model": "frida"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ClickHouse error: {e}")


@router.get("/info")
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
