from sentence_transformers import SentenceTransformer
from app.infrastructure.cache.redis_cache import RedisCacheService
from app.infrastructure.cache.semantic_cache import SemanticCacheService
from app.infrastructure.adapters.rag_adapter import rag_service

_embedding_model = None
# We use a dictionary to keep track of each instance with its respective namespace
_semantic_cache_instances = {}


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
    return _embedding_model


def get_semantic_cache_service(namespace: str = "general") -> SemanticCacheService:
    global _semantic_cache_instances

    if namespace not in _semantic_cache_instances:
        redis_cache = RedisCacheService()
        qdrant_client = rag_service.qdrant_client

        _semantic_cache_instances[namespace] = SemanticCacheService(
            redis_cache=redis_cache,
            qdrant_client=qdrant_client,
            embedding_model=get_embedding_model(),
            namespace=namespace
        )

    return _semantic_cache_instances[namespace]