"""
Custom exceptions for the headless trust & safety pipeline.

Defines error types that map to HTTP status codes so the API returns
meaningful errors instead of crashing.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base exception for application errors."""
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InvalidTextError(AppError):
    """Raised when input text is empty or invalid."""
    status_code = 422
    code = "invalid_text"


class TextTooLongError(AppError):
    """Raised when input text exceeds the maximum allowed length."""
    status_code = 413
    code = "text_too_long"


class AuditWriteError(AppError):
    """Raised when writing to the database fails."""
    status_code = 503
    code = "audit_unavailable"


class ModelUnavailableError(AppError):
    """Raised when the toxicity model cannot be loaded."""
    status_code = 503
    code = "model_unavailable"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Convert AppError into a JSON response with the right status code."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.code,
            "message": exc.message,
            "request_id": getattr(request.state, "request_id", None),
        },
    )
