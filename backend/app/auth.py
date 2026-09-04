"""
Authentication and Authorization module.

Provides JWT-based authentication for API and WebSocket endpoints.
Supports development/demo mode for easy local testing.
"""
from __future__ import annotations
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status, WebSocket, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.db.database import SessionLocal, User

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

# Demo/Development mode - allows unauthenticated access for local testing
DEMO_MODE = os.environ.get("DEMO_MODE", "true").lower() == "true"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[int]:
    """
    Extract user ID from JWT token.
    Returns None if in demo mode or token is invalid.
    """
    if DEMO_MODE:
        # In demo mode, return a default demo user ID
        return 1
    
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = decode_token(credentials.credentials)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return int(user_id)


def get_current_user_ws(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
) -> Optional[int]:
    """
    Extract user ID from JWT token for WebSocket connections.
    Supports token in query parameter or Authorization header.
    Returns None if in demo mode.
    """
    if DEMO_MODE:
        return 1
    
    # Try query parameter first
    auth_token = token
    
    # Fallback to Authorization header
    if auth_token is None:
        auth_header = websocket.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]
    
    if auth_token is None:
        return None
    
    payload = decode_token(auth_token)
    if payload is None or payload.get("type") != "access":
        return None
    
    user_id = payload.get("sub")
    if user_id is None:
        return None
    
    return int(user_id)


def require_admin(user_id: int = Depends(get_current_user_id)) -> int:
    """Dependency that requires admin privileges."""
    if DEMO_MODE:
        return user_id
    
    with SessionLocal() as session:
        user = session.get(User, user_id)
        if user is None or not user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
    return user_id


def create_demo_user_if_not_exists() -> None:
    """Create a default demo user for development mode."""
    if not DEMO_MODE:
        return
    
    with SessionLocal() as session:
        existing = session.query(User).filter(User.username == "demo").first()
        if existing is None:
            demo_user = User(
                username="demo",
                email="demo@voiceguard.local",
                hashed_password=get_password_hash("demo"),
                full_name="Demo User",
                is_active=True,
                is_admin=True,
            )
            session.add(demo_user)
            session.commit()
            logger.info("Created demo user")


# Export for use in other modules
__all__ = [
    "User",
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "get_current_user_id",
    "get_current_user_ws",
    "require_admin",
    "create_demo_user_if_not_exists",
    "DEMO_MODE",
    "security",
]