"""Application configuration."""

import os
from typing import Optional


class Settings:
    """Application settings from environment variables."""

    app_name: str = "Snake Detector"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://snake:snake_pass@localhost:5432/snake_detector"
    )
    
    # OCI
    oci_bucket: str = os.getenv("OCI_BUCKET", "snake-analytics-bucket")
    oci_namespace: str = os.getenv("OCI_NAMESPACE", "axynib9dtl3l")
    
    # Model
    model_path: str = os.getenv("MODEL_PATH", "/app/models/best.xml")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    
    # Server
    max_frame_size_mb: int = int(os.getenv("MAX_FRAME_SIZE_MB", "5"))
    upload_dir: str = os.getenv("UPLOAD_DIR", "/app/uploads")


settings = Settings()
