"""Snake Detector API - Main Application"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import engine, Base
from app.routes import inference, health

import logging

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting Snake Detector API...")
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")
    yield
    # Shutdown
    await engine.dispose()
    logger.info("Snake Detector API stopped")


app = FastAPI(
    title="Snake Detector API",
    description="Real-time corn snake detection and morph classification",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router)
app.include_router(inference.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "Snake Detector API",
        "version": "0.1.0",
        "status": "operational",
    }
