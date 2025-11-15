import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI
from clickhouse_client import ch
from routes.health_info import router as health_info_router
from routes.base import router as base_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)
load_dotenv()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Код при запуске
    # create_table_if_not_exists()
    yield
    # Код при завершении
    if ch:
        ch.close()
        logger.info("ClickHouse connection closed.")

app = FastAPI(
    title="FRIDA + ClickHouse RAG API (Manual)",
    lifespan=lifespan
)

app.include_router(health_info_router)
app.include_router(base_router)


# -------------------------------
# Uvicorn entry point
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Uvicorn server stopped.")
