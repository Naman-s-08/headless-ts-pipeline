"""
Main FastAPI application entry point.

Sets up lifespan events (load model, connect DB), middleware (request ID),
exception handlers, and includes the API router.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool

import app.api as api_module
from app.config import get_settings
from app.db import AuditRepository
from app.exceptions import AppError, app_error_handler
from app.logging_config import request_id_var, setup_logging
from app.model import ToxicityModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect database on startup, clean up on shutdown."""
    setup_logging(get_settings().log_level)

    # Load the toxicity model once (CPU-bound, run in threadpool)
    m = ToxicityModel()
    await run_in_threadpool(m.load)

    # Connect to the database
    r = AuditRepository()
    await r.connect()
    await r.ensure_indexes()

    # Share with api.py
    api_module.model = m
    api_module.repo = r

    logging.getLogger(__name__).info("startup complete")
    yield

    # Shutdown
    await r.close()


app = FastAPI(title="Headless Trust & Safety Pipeline", version="0.1.0", lifespan=lifespan)
app.add_exception_handler(AppError, app_error_handler)
app.include_router(api_module.router)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add a unique request ID to every request for tracing."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response
