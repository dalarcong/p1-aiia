"""
Connection to upload the images to the container at Azure
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
)
from dotenv import load_dotenv


load_dotenv()

CONNECTION_STRING = os.environ[
    "AZURE_STORAGE_CONNECTION_STRING"
]
CONTAINER_NAME = os.getenv(
    "AZURE_RAW_CONTAINER",
    "documents",
)
BLOB_PREFIX = os.getenv(
    "AZURE_RAW_PREFIX",
    "in",
)


_blob_service = BlobServiceClient.from_connection_string(
    CONNECTION_STRING
)

_container = _blob_service.get_container_client(
    CONTAINER_NAME
)


def upload_frame(
    jpeg: bytes,
    *,
    camera_id: str,
    captured_at: datetime,
) -> str:
    """Upload one JPEG frame to Azure Blob Storage."""

    timestamp = captured_at.strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    blob_name = (
        f"{BLOB_PREFIX}/"
        f"{camera_id}_{timestamp}.jpg"
    )

    _container.upload_blob(
        name=blob_name,
        data=jpeg,
        overwrite=False,
        content_settings=ContentSettings(
            content_type="image/jpeg"
        ),
    )

    return blob_name