"""SQLAlchemy models for inference results."""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, JSON
from app.core.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_url = Column(String(512), nullable=True)
    species = Column(String(100), nullable=False, default="Pantherophis guttatus")
    morph = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=False)
    bbox_coordinates = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    source = Column(String(50), nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "image_url": self.image_url,
            "species": self.species,
            "morph": self.morph,
            "confidence": self.confidence,
            "bbox": self.bbox_coordinates,
            "timestamp": self.created_at.isoformat() if self.created_at else None,
            "source": self.source,
        }
