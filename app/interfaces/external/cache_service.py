from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class CacheServiceInterface(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
        """Save data to cache with a Time-To-Live (TTL)."""
        pass
