"""
Tests for Phase 7 Enterprise Security & Governance.
Ensures API Keys, RBAC, PII Masking, and Audit logs work correctly.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app
from core.security.masking import get_pii_masker
from core.audit.logger import get_audit_logger

client = TestClient(app)

# ------------------------------------------------------------------ #
# API Key & RBAC Tests
# ------------------------------------------------------------------ #

def test_unauthenticated_access_denied():
    """Verify that hitting a secured endpoint without an API key returns 401."""
    # /models requires authentication
    response = client.get("/models")
    assert response.status_code == 401
    assert "Missing API Key" in response.json()["detail"]

def test_authenticated_access_success():
    """Verify that hitting a secured endpoint with a valid key works."""
    response = client.get("/models", headers={"X-API-Key": "sk-user-key-456"})
    assert response.status_code == 200
    assert "vision_models" in response.json()

def test_rbac_admin_required():
    """Verify that a USER role cannot hit an ADMIN endpoint."""
    # /rag/stats requires getting user, wait, /rag/stats is USER, /rag/reset is ADMIN.
    response = client.delete("/rag/reset", headers={"X-API-Key": "sk-user-key-456"})
    assert response.status_code == 403
    assert "requires UserRole.ADMIN" in response.json()["detail"]

def test_rbac_admin_success():
    """Verify that an ADMIN role CAN hit an ADMIN endpoint."""
    # Mock /rag/reset success or internal error if ChromaDB isn't setup, but not 403.
    response = client.delete("/rag/reset", headers={"X-API-Key": "sk-admin-secret-key-123"})
    assert response.status_code in (200, 500) # Depends if Chroma is up, but NOT 403

# ------------------------------------------------------------------ #
# Data Masking Tests
# ------------------------------------------------------------------ #

def test_pii_masker_redacts_ssn():
    masker = get_pii_masker()
    prompt = "My SSN is 123-45-6789. What do you see?"
    redacted, stats = masker.redact(prompt)
    assert redacted == "My SSN is [REDACTED_SSN]. What do you see?"
    assert stats["SSN"] == 1

def test_pii_masker_redacts_multiple():
    masker = get_pii_masker()
    prompt = "Email me at user@enterprise.com or call 555-123-4567. Credit card 1234-5678-9012-3456."
    redacted, stats = masker.redact(prompt)
    assert "user@enterprise.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_CREDIT_CARD]" in redacted
    assert stats["EMAIL"] == 1
    assert stats["PHONE"] == 1
    assert stats["CREDIT_CARD"] == 1

# ------------------------------------------------------------------ #
# Audit Log Tests
# ------------------------------------------------------------------ #

def test_audit_logger_records_query():
    logger = get_audit_logger()
    logger.log_request(
        username="test_admin",
        endpoint="/test/endpoint",
        prompt="[REDACTED_SSN]",
        status_code=200,
        response_meta={"dummy": "data"}
    )
    # Query latest
    logs = logger.query_logs(limit=1)
    assert len(logs) == 1
    assert logs[0]["username"] == "test_admin"
    assert logs[0]["endpoint"] == "/test/endpoint"
    assert logs[0]["prompt"] == "[REDACTED_SSN]"
    assert "dummy" in logs[0]["response_meta"]
