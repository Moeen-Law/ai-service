import os
import json
import redis.asyncio as redis
from typing import Any, Dict, Optional
from app.interfaces.external.cache_service import CacheServiceInterface
from app.infrastructure.logging.logger import get_logger

logger = get_logger(__name__)


class RedisCacheService(CacheServiceInterface):

    def __init__(self, redis_url: Optional[str] = None):
        # Read the Redis URL from the .env file; if not provided, fall back to the local Redis instance as the default.
        final_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/1")

        # Configure Redis connection settings.
        connection_kwargs = {"decode_responses": True}

        # If the URL uses a secure Redis connection (hosted server), add SSL certificate verification bypass settings.
        if final_url.startswith("rediss://"):
            connection_kwargs["ssl_cert_reqs"] = "none"

        self._client = redis.from_url(final_url, **connection_kwargs)

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            data = await self._client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl_seconds: int = 2592000) -> None:
        try:
            data_str = json.dumps(value, ensure_ascii=False)
            await self._client.setex(name=key, time=ttl_seconds, value=data_str)
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")