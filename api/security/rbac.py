"""
Sentinel AI — Role Based Access Control (RBAC)
Provides dependency generators to restrict endpoints based on user roles.
"""
from fastapi import Depends, HTTPException, status
from api.security.auth import get_current_user, UserRole, UserAccount

def require_role(required_role: UserRole):
    """
    Returns a FastAPI dependency that checks the current user's role.
    Usage: user: UserAccount = Depends(require_role(UserRole.ADMIN))
    """
    async def role_checker(user: UserAccount = Depends(get_current_user)) -> UserAccount:
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
        
        # Roles hierarchy check (ADMIN overrides)
        if user.role != required_role and user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires {required_role} role. Current: {user.role}"
            )
        
        return user
        
    return role_checker
