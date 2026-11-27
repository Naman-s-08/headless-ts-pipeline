"""
API routes for the headless trust & safety pipeline.

Defines:
- GET /health — check if the service and model are running
- POST /moderate — moderate a single text comment
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.db import AuditRepository
from app.exceptions import (
    AppError,
    InvalidTextError,
    TextTooLongError,
    app_error_handler,
)
from app.logging_config import request_id_var, setup_logging
from app.model import ModelLoadError, ToxicityModel
from app.policy import decide
from app.preprocess import clean_text
from app.schemas import AuditLog, ModerationRequest, ModerationResponse

router = APIRouter()

# These are set during app startup (see app/main.py)
model: ToxicityModel | None = None
repo: AuditRepository | None = None


@router.get("/health")
async def health(request: Request):
    """Health check endpoint.

    Returns whether the service is up and whether the toxicity model is loaded.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None and model._pipe is not None,
    }


@router.post("/moderate", response_model=ModerationResponse)
async def moderate(body: ModerationRequest, request: Request):
    """
    Moderate a single text comment.

    Pipeline:
        1. Validate text length
        2. Clean the text (preprocess)
        3. Score toxicity (model)
        4. Decide ALLOW/REVIEW/BLOCK (policy)
        5. Log to database (db)
        6. Return response
    """
    settings = get_settings()
    started = time.perf_counter()

    # Step 1: Validate text length
    if len(body.text) > settings.max_text_chars:
        raise TextTooLongError(f"text exceeds {settings.max_text_chars} characters")

    # Step 2: Clean the text
    try:
        cleaned = clean_text(body.text)
    except ValueError as exc:
        raise InvalidTextError(str(exc)) from exc

    # Step 3: Score toxicity (CPU-bound, run in threadpool)
    if model is None:
        raise ModelUnavailableError("toxicity model is not loaded")
    try:
        label, score = await run_in_threadpool(model.predict, cleaned)
    except Exception as exc:
        raise ModelUnavailableError(f"model inference failed: {exc}") from exc

    # Step 4: Decide ALLOW / REVIEW / BLOCK
    action = decide(
        score,
        settings.toxicity_allow_threshold,
        settings.toxicity_review_threshold,
    )

    # Step 5: Log to database
    elapsed_ms = (time.perf_counter() - started) * 1000
    if repo is not None:
        try:
            await repo.log(
                AuditLog(
                    text=body.text,
                    cleaned_text=cleaned,
                    label=label,
                    score=score,
                    action=action,
                    timestamp_utc=datetime.now(timezone.utc),
                    model_id=settings.model_id,
                    processing_time_ms=elapsed_ms,
                    client_ip=request.client.host if request.client else None,
                )
            )
        except Exception as exc:
            raise AuditWriteError(f"failed to write audit log: {exc}") from exc

    # Step 6: Return response
    return ModerationResponse(
        request_id=request.state.request_id,
        action=action,
        label=label,
        score=score,
        processing_time_ms=elapsed_ms,
    )
