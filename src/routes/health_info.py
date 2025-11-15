from fastapi import APIRouter, Depends, HTTPException
from clickhouse_client import ch
import logging
from tools import get_api_key_dependency

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check():
    if not ch:
        raise HTTPException(
            status_code=500, detail="ClickHouse client not initialized."
        )
    try:
        ch.ping()
        return {
            "status": "healthy",
            "clickhouse": "connected",
            "embedding_model": "frida",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ClickHouse error: {e}")


@router.get("/stats")
async def get_stats(api_key: str = Depends(get_api_key_dependency)):
    """Получение статистики по данным в таблице."""
    try:
        stats_query = """
        SELECT
            count() as total_chunks,
            uniqExact(id) as unique_ids,
            min(created_at) as oldest_record,
            max(created_at) as newest_record
        FROM novaya
        """

        result = ch.query(stats_query)

        if result.result_rows:
            total_chunks, unique_ids, oldest, newest = result.result_rows[0]
            return {
                "total_chunks": total_chunks,
                "unique_ids": unique_ids,
                "oldest_record": oldest.isoformat() if oldest else None,
                "newest_record": newest.isoformat() if newest else None,
                "table": "novaya",
            }
        else:
            return {"error": "No data available"}

    except Exception as e:
        logger.exception("❌ Ошибка при получении статистики")
        raise HTTPException(status_code=500, detail=str(e))
