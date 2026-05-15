"""Inference routes - image upload and real-time detection."""

import os
import uuid
import logging
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

os.makedirs(settings.upload_dir, exist_ok=True)

# Model paths
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_XML = MODEL_DIR / "best.xml"
MODEL_BIN = MODEL_DIR / "best.bin"
MODEL_PT = MODEL_DIR / "best.pt"

_model = None


def get_model():
    """Lazy-load model with fallback priority: OpenVINO > PyTorch > None"""
    global _model
    if _model is not None:
        return _model
    
    try:
        if MODEL_XML.exists() and MODEL_BIN.exists():
            logger.info("Loading OpenVINO model...")
            from ultralytics import YOLO
            _model = YOLO(str(MODEL_XML))
            logger.info("✓ OpenVINO model loaded")
            return _model
    except Exception as e:
        logger.warning(f"OpenVINO load failed: {e}")
    
    try:
        if MODEL_PT.exists():
            logger.info("Loading PyTorch model...")
            from ultralytics import YOLO
            _model = YOLO(str(MODEL_PT))
            logger.info("✓ PyTorch model loaded")
            return _model
    except Exception as e:
        logger.warning(f"PyTorch load failed: {e}")
    
    logger.warning("No model found - detection will return placeholder results")
    return None


@router.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """Upload an image for snake morph detection."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files supported")
    
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    file_path = os.path.join(settings.upload_dir, f"{file_id}{ext}")
    
    content = await file.read()
    if len(content) > settings.max_frame_size_mb * 1024 * 1024:
        raise HTTPException(400, f"Max {settings.max_frame_size_mb}MB")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    model = get_model()
    if model is None:
        return {
            "id": file_id,
            "status": "no_model",
            "message": "Model not available. Train first.",
            "detections": [],
        }
    
    results = model(file_path, conf=settings.confidence_threshold)
    
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                })
    
    return {
        "id": file_id,
        "status": "completed",
        "detections": detections,
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time frame processing."""
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            await websocket.send_json({
                "status": "received",
                "size_bytes": len(data),
            })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
