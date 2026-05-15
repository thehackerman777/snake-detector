"""Inference routes - image upload and real-time detection."""

import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Ensure upload directory exists
os.makedirs(settings.upload_dir, exist_ok=True)

# Model cache (loaded on first request)
_model = None


def get_model():
    """Lazy-load the YOLO model (OpenVINO format)."""
    global _model
    if _model is None:
        model_path = settings.model_path
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Detection disabled.")
            return None
        try:
            from ultralytics import YOLO
            _model = YOLO(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    return _model


@router.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    Upload an image for snake detection.
    Returns detected species, morph, and confidence.
    """
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported")
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    file_path = os.path.join(settings.upload_dir, f"{file_id}{ext}")
    
    content = await file.read()
    if len(content) > settings.max_frame_size_mb * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {settings.max_frame_size_mb}MB)")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Run inference
    model = get_model()
    if model is None:
        # Placeholder response until model is trained
        return {
            "id": file_id,
            "status": "no_model",
            "message": "Model not yet trained. Upload received.",
            "file": file_path,
            "detections": [],
        }
    
    results = model(file_path, conf=settings.confidence_threshold)
    
    # Parse results
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            })
    
    return {
        "id": file_id,
        "status": "completed",
        "detections": detections,
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video frame processing.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Receive frame as bytes
            data = await websocket.receive_bytes()
            
            # TODO: Process frame through model
            # For now, acknowledge receipt
            await websocket.send_json({
                "status": "received",
                "size_bytes": len(data),
            })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011)
