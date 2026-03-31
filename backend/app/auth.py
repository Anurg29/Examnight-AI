"""
auth.py — User registration, login, and JWT verification.

Endpoints:
  POST /api/auth/register   { username, password }  → 201 { message }
  POST /api/auth/login      { username, password }  → 200 { access_token, token_type, username }
  GET  /api/auth/me         (Authorization: Bearer <token>) → 200 { username }
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import JWT_EXPIRE_HOURS, JWT_SECRET, MONGODB_URI

# ── Helpers ───────────────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
bearer_scheme = HTTPBearer(auto_error=False)
router = APIRouter(prefix='/api/auth', tags=['auth'])

ALGORITHM = 'HS256'

# Lazy MongoDB client — created once on first use
_mongo_client: AsyncIOMotorClient | None = None


def _get_db():
    global _mongo_client
    if not MONGODB_URI:
        raise RuntimeError('MONGODB_URI is not configured. Add it to your .env or Render env vars.')
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(MONGODB_URI)
    return _mongo_client.get_default_database()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    username: str


class MeResponse(BaseModel):
    username: str


# ── JWT helpers ───────────────────────────────────────────────────────────────

def _create_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode({'sub': username, 'exp': expire}, JWT_SECRET, algorithm=ALGORITHM)


def _decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        username: str | None = payload.get('sub')
        if not username:
            raise ValueError
        return username
    except (JWTError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid or expired token.',
            headers={'WWW-Authenticate': 'Bearer'},
        ) from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing Bearer token.',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    return _decode_token(credentials.credentials)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post('/register', status_code=status.HTTP_201_CREATED)
async def register(body: AuthRequest):
    """Create a new user account. Usernames are stored in lower-case."""
    username = body.username.strip().lower()
    if not username or not body.password:
        raise HTTPException(status_code=400, detail='Username and password are required.')

    db = _get_db()
    existing = await db['users'].find_one({'username': username})
    if existing:
        raise HTTPException(status_code=409, detail='Username already taken.')

    hashed = pwd_context.hash(body.password)
    await db['users'].insert_one({
        'username': username,
        'password_hash': hashed,
        'created_at': datetime.now(timezone.utc).isoformat(),
    })
    return {'message': f'Account created for {username}.'}


@router.post('/login', response_model=TokenResponse)
async def login(body: AuthRequest):
    """Verify credentials and return a JWT access token."""
    username = body.username.strip().lower()
    db = _get_db()
    user = await db['users'].find_one({'username': username})

    if not user or not pwd_context.verify(body.password, user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username or password.',
        )

    token = _create_token(username)
    return TokenResponse(access_token=token, username=username)


@router.get('/me', response_model=MeResponse)
async def me(current_user: str = Depends(get_current_user)):
    """Return the username of the currently authenticated user."""
    return MeResponse(username=current_user)
