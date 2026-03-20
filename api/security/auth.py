"""
Sentinel AI — API Authentication Layer
Provides API Key extraction, verification, and user management.
"""
import os
from enum import Enum
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# In a real enterprise system, these would be loaded from a DB or Identity Provider (Vault/AWS KMS)
# For Phase 7, we'll mock them heavily with environment variables/hardcoded keys for the MVP.
class UserRole(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"
    ENGINEER = "ENGINEER"

class UserAccount(BaseModel):
    username: str
    role: UserRole
    api_key: str

# Mock Database of valid API Keys
_MOCK_ACCOUNTS = {
    os.getenv("SENTINEL_ADMIN_KEY", "sk-admin-secret-key-123"): UserAccount(
        username="admin_service", role=UserRole.ADMIN, api_key="sk-admin-secret-key-123"
    ),
    os.getenv("SENTINEL_USER_KEY", "sk-user-key-456"): UserAccount(
        username="standard_user", role=UserRole.USER, api_key="sk-user-key-456"
    ),
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(api_key_header: str = Security(api_key_header)) -> UserAccount:
    """Validate API Key and lookup user account. Used as FastAPI dependency."""
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key in X-API-Key header",
        )
    
    user = _MOCK_ACCOUNTS.get(api_key_header)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return user
