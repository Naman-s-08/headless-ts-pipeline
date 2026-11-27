"""
Database audit logging layer.

This is a stub that stores audit logs in memory. Later you can replace
this with Motor (MongoDB async driver) using the same method names.
"""

from datetime import datetime, timezone
from typing import Optional

from app.schemas import AuditLog


class AuditRepository:
    """Stores moderation audit records."""

    def __init__(self) -> None:
        self.records: list[AuditLog] = []

    async def connect(self) -> None:
        """Placeholder for database connection."""
        pass

    async def ensure_indexes(self) -> None:
        """Placeholder for index creation."""
        pass

    async def log(self, record: AuditLog) -> str:
        """Save an audit record and return its ID."""
        self.records.append(record)
        return str(len(self.records))

    async def close(self) -> None:
        """Placeholder for database cleanup."""
        pass
