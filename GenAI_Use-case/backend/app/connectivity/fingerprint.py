"""Stable source identity (dialect + host + database + schema) so repeat visits find prior state.

The same source must always produce the same id, whoever connects. So the username and password
are deliberately excluded: two stewards with different logins are still looking at the same data.
"""
import hashlib
from pathlib import Path

from app.connectivity.connectors import ConnectionConfig


def source_fingerprint(config: ConnectionConfig) -> str:
    if config.dialect == "sqlite":
        database = str(Path(config.database).resolve()).lower()
    else:
        database = config.database.lower()
    parts = [
        config.dialect,
        (config.host or "").lower(),
        str(config.port or ""),
        database,
        (config.schema_name or "").lower(),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
