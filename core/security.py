import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from config import settings

security = HTTPBearer()

class AuthenticatedUser(BaseModel):
    id: str
    role: str | None = None
    department: str | None = None
    level: int | None = None

def decode_access_token(token: str, secret: str = settings.JWT_SECRET) -> AuthenticatedUser:
    """Decode and validate JWT access token using shared JWT secret."""
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        user_id = payload.get("userId") or payload.get("sub") or payload.get("id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing userId",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        level_val = payload.get("level")
        parsed_level = int(level_val) if level_val is not None and str(level_val).isdigit() else None

        return AuthenticatedUser(
            id=str(user_id),
            role=payload.get("role"),
            department=payload.get("department"),
            level=parsed_level,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token signature has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> AuthenticatedUser:
    """FastAPI Dependency for validating bearer JWT tokens."""
    return decode_access_token(credentials.credentials)
