"""Content hashing for idempotent knowledge ingestion."""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of UTF-8 text.

    Used for document and chunk ``content_hash`` fields so unchanged material
    can be skipped and changed material can be replaced cleanly.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
