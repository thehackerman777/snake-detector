"""OCI Object Storage service."""

import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class ObjectStorageService:
    """Service for uploading images to Oracle Object Storage."""
    
    def __init__(self):
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the OCI Object Storage client."""
        try:
            import oci
            self.client = oci.object_storage.ObjectStorageClient(
                config=oci.config.from_file()
            )
            logger.info("OCI Object Storage client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize OCI client: {e}")
    
    def upload_image(self, file_path: str, object_name: Optional[str] = None) -> Optional[str]:
        """
        Upload an image to Object Storage.
        Returns the public URL or None on failure.
        """
        if not self.client:
            logger.warning("OCI client not available")
            return None
        
        try:
            object_name = object_name or file_path.split("/")[-1]
            
            with open(file_path, "rb") as f:
                self.client.put_object(
                    namespace_name=settings.oci_namespace,
                    bucket_name=settings.oci_bucket,
                    object_name=object_name,
                    put_object_body=f,
                )
            
            url = f"https://objectstorage.{settings.oci_namespace}.oraclecloud.com/n/{settings.oci_namespace}/b/{settings.oci_bucket}/o/{object_name}"
            logger.info(f"Image uploaded: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to upload to OCI: {e}")
            return None


storage_service = ObjectStorageService()
