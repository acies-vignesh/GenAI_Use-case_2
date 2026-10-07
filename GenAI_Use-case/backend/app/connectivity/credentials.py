"""Encrypt/decrypt stored connection secrets (never persist plaintext passwords).

Fernet = AES-128 encryption + an integrity check, keyed by ENCRYPTION_KEY from .env. Losing that
key means stored passwords can't be decrypted and must be entered again - that is the point.
"""
from cryptography.fernet import Fernet, InvalidToken

from app.config import settings


class CredentialError(RuntimeError):
    pass


def _fernet(key: str | None = None) -> Fernet:
    key = key or settings.encryption_key
    if not key:
        raise CredentialError("ENCRYPTION_KEY is not set in .env - cannot store passwords")
    return Fernet(key.encode())


def encrypt(secret: str, key: str | None = None) -> str:
    return _fernet(key).encrypt(secret.encode()).decode()


def decrypt(token: str, key: str | None = None) -> str:
    try:
        return _fernet(key).decrypt(token.encode()).decode()
    except InvalidToken as e:
        raise CredentialError("Stored password can't be decrypted (ENCRYPTION_KEY changed?) - "
                              "re-enter the password") from e
