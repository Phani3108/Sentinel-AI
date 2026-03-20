"""
Sentinel AI — Distributed Caching Layer
Phase 8: High Availability & Scaling

Provides exact-match hash caching for the entire multimodal pipeline.
If the exact same image hash + identical prompt arrives, this bypassed the GPU entirely.
"""
import json
import logging
import hashlib
from typing import Optional, Dict, Any

try:
    import redis
    redis_available = True
except ImportError:
    redis_available = False

logger = logging.getLogger(__name__)

class SentinelCache:
    """Redis singleton for inference result caching."""

    _instance = None

    def __new__(cls, redis_url: str = "redis://localhost:6379/0"):
        if cls._instance is None:
            cls._instance = super(SentinelCache, cls).__new__(cls)
            cls._instance._init(redis_url)
        return cls._instance

    def _init(self, redis_url: str):
        self.enabled = False
        if redis_available:
            try:
                self.client = redis.Redis.from_url(redis_url, decode_responses=True)
                # Test connection
                self.client.ping()
                self.enabled = True
                logger.info("✅ Redis Cache Connected.")
            except Exception as e:
                logger.warning(f"Redis cache disabled (Connection failed: {e})")
        else:
            logger.warning("Redis cache disabled (`redis` package not installed).")

    def _generate_key(self, image_hash: str, prompt: str) -> str:
        """Create a deterministic cache key from the image and prompt."""
        combo = f"{image_hash}::{prompt}"
        return "sentinel:inference:" + hashlib.sha256(combo.encode()).hexdigest()

    def get_inference(self, image_hash: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Attempt to retrieve a cached result."""
        if not self.enabled:
            return None
            
        key = self._generate_key(image_hash, prompt)
        try:
            val = self.client.get(key)
            if val:
                logger.info(f"⚡ Redis Cache HIT for prompt: '{prompt}'")
                return json.loads(val)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
        return None

    def set_inference(self, image_hash: str, prompt: str, result_dict: Dict[str, Any], ttl_seconds: int = 86400):
        """Cache an exact-match result for 24 hours by default."""
        if not self.enabled:
            return
            
        key = self._generate_key(image_hash, prompt)
        try:
            self.client.setex(key, ttl_seconds, json.dumps(result_dict))
            logger.info("💾 Cached inference result in Redis.")
        except Exception as e:
            logger.error(f"Redis SET error: {e}")

_cache_instance = None

def get_cache() -> SentinelCache:
    global _cache_instance
    if _cache_instance is None:
        import os
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _cache_instance = SentinelCache(redis_url=redis_url)
    return _cache_instance
