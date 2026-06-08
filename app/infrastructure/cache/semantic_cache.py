import hashlib
import uuid
from typing import Any, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

from app.interfaces.external.cache_service import CacheServiceInterface
from app.infrastructure.logging.logger import get_logger

logger = get_logger(__name__)


class SemanticCacheService(CacheServiceInterface):
    def __init__(
            self,
            redis_cache: CacheServiceInterface,
            qdrant_client: QdrantClient,
            embedding_model: SentenceTransformer,
            collection_name: str = "semantic_cache_collection",
            similarity_threshold: float = 0.92,
            namespace: str = "general" # Added namespace support to isolate cache entries per workflow.
    ):
        self._redis = redis_cache
        self._qdrant = qdrant_client
        self._model = embedding_model
        self._collection_name = collection_name
        self._threshold = similarity_threshold
        self._namespace = namespace

    def initialize_collection(self):
        exists = self._qdrant.collection_exists(self._collection_name)
        if not exists:
            self._qdrant.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self._collection_name}")

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            query_text = f"query: {key.strip()}"
            vector = self._model.encode(query_text).tolist()

            response = self._qdrant.query_points(
                collection_name=self._collection_name,
                query=vector,
                limit=1,
                score_threshold=self._threshold,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="namespace",
                            match=models.MatchValue(value=self._namespace)
                        )
                    ]
                ),
                with_payload=True
            )

            hits = response.points
            if not hits:
                return None

            best_match = hits[0]
            redis_key = best_match.payload.get("redis_key")

            if not redis_key:
                logger.info(f"semantic_cache_qdrant_hit_but_no_redis_key [{self._namespace}] for query: {key}")
                return None

            result = await self._redis.get(redis_key)
            if result is None:
                logger.info(f"semantic_cache_qdrant_hit_but_redis_expired [{self._namespace}] for query: {key}")
                try:
                    self._qdrant.delete(
                        collection_name=self._collection_name,
                        points_selector=models.PointIdsList(points=[best_match.id]),
                    )
                    logger.info(f"semantic_cache_qdrant_vector_deleted [{self._namespace}] id: {best_match.id}")
                except Exception as del_exc:
                    logger.warning(f"semantic_cache_qdrant_delete_failed [{self._namespace}]: {del_exc}")
                return None

            logger.info(f"semantic_cache_hit [{self._namespace}] for query: {key}")
            return result
        except Exception as e:
            logger.warning(f"Semantic Cache GET failed: {e}")
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl_seconds: int = 2592000) -> None:
        try:
            clean_query = key.strip().lower()
            if len(key.split()) < 3:
                logger.info(f"Skipping cache for short query: {key}")
                return
            query_hash = hashlib.sha256(clean_query.encode('utf-8')).hexdigest()

            # redis key depends on the namespace
            redis_key = f"semantic_{self._namespace}:{query_hash}"

            await self._redis.set(redis_key, value, ttl_seconds)

            query_text = f"query: {clean_query}"
            vector = self._model.encode(query_text).tolist()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, redis_key))

            self._qdrant.upsert(
                collection_name=self._collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "redis_key": redis_key,
                            "original_query": key,
                            "namespace": self._namespace
                        }
                    )
                ]
            )
            logger.info(f"semantic_cache_saved [{self._namespace}] for query: {key}")

        except Exception as e:
            logger.warning(f"Semantic Cache SET failed: {e}")