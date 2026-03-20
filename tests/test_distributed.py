"""
Tests for Phase 8 Distributed Scaling & HA
Validates Redis caching, Celery async task dispatching, and FastAPI job polling.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from core.cache import get_cache

client = TestClient(app)

# ------------------------------------------------------------------ #
# Redis Cache Tests
# ------------------------------------------------------------------ #

def test_redis_cache_singleton_logic():
    """Verify the Redis cache singleton behaves correctly gracefully falling back if no connection."""
    cache = get_cache()
    # By default in CI without a Redis server, enabled might be False.
    # We will test the logic assuming we can inject data.
    
    # Store dummy data
    cache.set_inference(image_hash="abc1234", prompt="Test Prompt", result_dict={"answer": "Cached!"})
    
    # If redis isn't running locally during pytest, it shouldn't crash.
    result = cache.get_inference("abc1234", "Test Prompt")
    if cache.enabled:
        assert result["answer"] == "Cached!"
    else:
        assert result is None

@patch("core.cache.SentinelCache.get_inference")
def test_analyze_image_uses_cache(mock_get_inference):
    """Verify that the FastAPI endpoint queries the cache and returns early on HIT."""
    mock_get_inference.return_value = {
        "vision_model": "mock_vision",
        "vision_description": "from_cache",
        "final_answer": "This is a cached answer",
        "total_latency_ms": 1.0,
        "llm_tokens_used": 10,
        "rag_context": None,
        "rag_chunks_retrieved": 0,
        "cache": "HIT"
    }

    # Hit the endpoint using a dummy file (Auth required!)
    # Phase 7 secured this, so we need the valid API key
    headers = {"X-API-Key": "sk-user-key-456"}
    files = {"file": ("dummy.jpg", b"fake_image_bytes", "image/jpeg")}
    data = {"prompt": "What do you see in this image?"}
    
    response = client.post("/analyze/image", headers=headers, files=files, data=data)
    
    assert response.status_code == 200
    assert response.json()["final_answer"] == "This is a cached answer"
    mock_get_inference.assert_called_once()


# ------------------------------------------------------------------ #
# Celery Async Jobs Tests
# ------------------------------------------------------------------ #

@patch("api.jobs.celery_app")
def test_submit_video_job_dispatches_task(mock_celery):
    """Verify /jobs/video successfully queues a task via Celery."""
    # Mock the celery send_task response
    mock_task = MagicMock()
    mock_task.id = "mock-uuid-1234-5678"
    mock_celery.send_task.return_value = mock_task
    
    headers = {"X-API-Key": "sk-user-key-456"}
    files = {"file": ("dummy.mp4", b"fake_video_bytes", "video/mp4")}
    data = {"prompt": "What happens here?"}
    
    response = client.post("/jobs/video", headers=headers, files=files, data=data)
    
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["job_id"] == "mock-uuid-1234-5678"
    assert res_data["status"] == "PENDING"
    mock_celery.send_task.assert_called_once()


@patch("api.jobs.celery_app")
@patch("celery.result.AsyncResult")
def test_poll_job_status(mock_async_result, mock_celery):
    """Verify /jobs/{job_id} correctly polls celery."""
    mock_res = MagicMock()
    mock_res.status = "SUCCESS"
    mock_res.result = {"narration": "A cat jumps on a table."}
    mock_async_result.return_value = mock_res
    
    headers = {"X-API-Key": "sk-user-key-456"}
    
    response = client.get("/jobs/mock-uuid-1234-5678", headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"
    assert response.json()["result"]["narration"] == "A cat jumps on a table."
