import clickhouse_connect
import logging
import os

logger = logging.getLogger(__name__)

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


# ch = clickhouse_connect.get_client(host='localhost', port=8123)
