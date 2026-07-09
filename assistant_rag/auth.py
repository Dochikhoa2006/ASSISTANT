"""JWT authentication helpers for REST and WebSocket entry points."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Annotated, Any, Optional

from fastapi import Header, HTTPException

from .contracts import AuthContext


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


@dataclass(frozen=True)
class AuthSettings:
    issuer: str | None = None
    audience: str | None = None
    jwks_url: str | None = None
    public_key: str | None = None
    required_scopes: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "AuthSettings":
        required = os.getenv("AUTH_REQUIRED_SCOPES", "")
        return cls(
            issuer=os.getenv("AUTH_JWT_ISSUER") or None,
            audience=os.getenv("AUTH_JWT_AUDIENCE") or None,
            jwks_url=os.getenv("AUTH_JWKS_URL") or None,
            public_key=os.getenv("AUTH_JWT_PUBLIC_KEY") or None,
            required_scopes=tuple(scope.strip() for scope in required.split(",") if scope.strip()),
        )

    @property
    def verification_configured(self) -> bool:
        return bool(self.issuer or self.audience or self.jwks_url or self.public_key or self.required_scopes)


def _extract_scopes(claims: dict[str, Any]) -> tuple[str, ...]:
    scopes = claims.get("scopes")
    if isinstance(scopes, list):
        return tuple(str(scope) for scope in scopes)
    scope_string = claims.get("scope")
    if isinstance(scope_string, str):
        return tuple(scope for scope in scope_string.split() if scope)
    return ()


def _decode_hs256_without_pyjwt(token: str, secret: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid JWT")
    header = json.loads(_b64url_decode(parts[0]))
    if header.get("alg") != "HS256":
        raise HTTPException(status_code=401, detail="PyJWT is required for this JWT algorithm")
    signed = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url_decode(parts[2]), expected):
        raise HTTPException(status_code=401, detail="Invalid JWT signature")
    return json.loads(_b64url_decode(parts[1]))


def _validate_standard_claims(claims: dict[str, Any], settings: AuthSettings) -> AuthContext:
    subject = claims.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="JWT subject is required")
    now = int(time.time())
    exp = claims.get("exp")
    if exp is not None and int(exp) < now:
        raise HTTPException(status_code=401, detail="JWT has expired")
    if settings.issuer and claims.get("iss") != settings.issuer:
        raise HTTPException(status_code=401, detail="Invalid JWT issuer")
    if settings.audience:
        aud = claims.get("aud")
        audiences = aud if isinstance(aud, list) else [aud]
        if settings.audience not in audiences:
            raise HTTPException(status_code=401, detail="Invalid JWT audience")
    scopes = _extract_scopes(claims)
    missing = [scope for scope in settings.required_scopes if scope not in scopes]
    if missing:
        raise HTTPException(status_code=403, detail="Required scope is missing")
    return AuthContext(user_id=str(subject), claims=claims, scopes=scopes)


def authenticate_token(token: str, settings: AuthSettings | None = None) -> AuthContext:
    settings = settings or AuthSettings.from_env()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")

    if not settings.verification_configured:
        return AuthContext(user_id=token, claims={"sub": token, "legacy_dev_token": True}, scopes=())

    try:
        import jwt  # type: ignore
    except ImportError:
        if settings.public_key:
            return _validate_standard_claims(
                _decode_hs256_without_pyjwt(token, settings.public_key),
                settings,
            )
        raise HTTPException(status_code=500, detail="PyJWT[crypto] is required for JWT authentication")

    try:
        key: Any = settings.public_key
        if settings.jwks_url:
            jwks_client = jwt.PyJWKClient(settings.jwks_url)
            key = jwks_client.get_signing_key_from_jwt(token).key
        options = {
            "require": ["sub", "exp"],
            "verify_aud": bool(settings.audience),
            "verify_iss": bool(settings.issuer),
        }
        claims = jwt.decode(
            token,
            key=key,
            algorithms=["RS256", "ES256", "HS256"],
            issuer=settings.issuer,
            audience=settings.audience,
            options=options,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid JWT: {exc}") from exc
    return _validate_standard_claims(dict(claims), settings)


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = authorization[len("Bearer ") :].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")
    return token


async def get_auth_context(
    authorization: Annotated[Optional[str], Header()] = None,
) -> AuthContext:
    return authenticate_token(_bearer_token(authorization))


async def get_current_user(
    authorization: Annotated[Optional[str], Header()] = None,
) -> str:
    return (await get_auth_context(authorization)).user_id
