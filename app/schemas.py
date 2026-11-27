"""
Data schemas for the moderation pipeline.

Defines the shape of data that flows through:
- Request (what comes in from the client)
- Response (what goes back to the client)
- AuditLog (what gets saved to the database)
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Action(str, Enum):
    """Possible moderation actions."""
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class ModerationRequest(BaseModel):
    """Input from the client application."""
    text: str = Field(min_length=1, description="Raw user comment to moderate")


class ModerationResponse(BaseModel):
    """Output returned to the client."""
    request_id: str
    action: Action
    label: str
    score: float
    processing_time_ms: float


class AuditLog(BaseModel):
    """Record saved to the database for every moderation decision."""
    text: str
    cleaned_text: str
    label: str
    score: float
    action: Action
    timestamp_utc: datetime
    model_id: str
    processing_time_ms: float
    client_ip: Optional[str] = None
