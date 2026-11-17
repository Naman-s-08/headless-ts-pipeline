"""Main FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(title="headless-ts-pipeline", version="0.1.0")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "headless-ts-pipeline skeleton"}


@app.get("/health")
def health():
    return {"status": "healthy"}
