import pytest
import jwt
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException
from core.security import decode_access_token, AuthenticatedUser
from config import settings

SECRET = settings.JWT_SECRET

def test_valid_token_decodes_user_identity_and_claims():
    """Verify that a valid signed JWT correctly extracts user ID, role, department, and level."""
    payload = {
        "userId": "user-456",
        "role": "STUDENT",
        "department": "Computer Science",
        "level": 300,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    
    user = decode_access_token(token, secret=SECRET)
    assert isinstance(user, AuthenticatedUser)
    assert user.id == "user-456"
    assert user.role == "STUDENT"
    assert user.department == "Computer Science"
    assert user.level == 300

def test_expired_token_raises_401():
    """Verify that an expired JWT token is rejected with 401 Unauthorized."""
    payload = {
        "userId": "user-456",
        "exp": datetime.now(timezone.utc) - timedelta(seconds=10)
    }
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token, secret=SECRET)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()

def test_token_missing_user_id_raises_401():
    """Verify that a token missing the userId/sub claim is rejected with 401 Unauthorized."""
    payload = {
        "role": "STUDENT",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token, secret=SECRET)
    assert exc_info.value.status_code == 401
    assert "missing userid" in exc_info.value.detail.lower()

def test_token_with_invalid_signature_raises_401():
    """Verify that a token signed with a different secret is rejected."""
    payload = {
        "userId": "user-456",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
    }
    wrong_token = jwt.encode(payload, "different_secret_key", algorithm="HS256")
    
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(wrong_token, secret=SECRET)
    assert exc_info.value.status_code == 401
    assert "invalid jwt token" in exc_info.value.detail.lower()
