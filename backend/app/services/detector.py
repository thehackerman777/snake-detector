"""Detection service - abstraction layer for model inference."""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class DetectionResult:
    """Result of a detection inference."""
    
    def __init__(
        self,
        morph: str,
        confidence: float,
        bbox: List[float],
    ):
        self.morph = morph
        self.confidence = confidence
        self.bbox = bbox
    
    def to_dict(self) -> Dict:
        return {
            "morph": self.morph,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


class SnakeDetector:
    """Wrapper around YOLO model for snake detection."""
    
    # Known morph classes (will be expanded after training)
    MORPH_CLASSES = [
        "amelanistic",
        "anerythristic",
        "motley",
        "stripe",
        "tessera",
        "palmetto",
        "snow",
        "normal",
    ]
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
    
    def load_model(self):
        """Load or reload the model."""
        if not self.model_path:
            logger.warning("No model path configured")
            return False
        
        try:
            # Will load OpenVINO model when available
            # from ultralytics import YOLO
            # self.model = YOLO(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, image_path: str) -> List[DetectionResult]:
        """Run detection on an image."""
        if not self.model:
            return []
        
        # results = self.model(image_path)
        # Parse results...
        return []
    
    def predict_frame(self, frame_bytes: bytes) -> List[DetectionResult]:
        """Run detection on raw frame bytes."""
        # import cv2
        # import numpy as np
        # frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        # results = self.model(frame)
        return []


# Singleton instance
detector = SnakeDetector()
