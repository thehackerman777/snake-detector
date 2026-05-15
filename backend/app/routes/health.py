"""Health check endpoint."""

from fastapi import APIRouter
import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "service": "snake-detector-api",
    }
