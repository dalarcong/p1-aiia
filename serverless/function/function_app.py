import json
import logging
import os
import time
from datetime import datetime, timezone

import azure.functions as func
import cv2
import numpy as np

from pathlib import Path
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


app = func.FunctionApp()

VISION_CONFIDENCE = float(
    os.getenv("VISION_CONFIDENCE", "0.5")
)

_vision_client = None


def get_vision_client() -> ImageAnalysisClient:
    """Create the Azure AI Vision client lazily and reuse it."""

    global _vision_client

    if _vision_client is None:
        endpoint = os.environ["AZURE_AI_VISION_ENDPOINT"]
        key = os.environ["AZURE_AI_VISION_KEY"]

        _vision_client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    return _vision_client

def get_camera_id(blob_name: str) -> str:
    filename = Path(blob_name).name
    parts = filename.split("__")

    if len(parts) >= 2:
        return parts[1]

    return "unknown"

def get_exclude_zone():
    """Parse the optional exclusion polygon from an environment variable."""

    value = os.getenv("EVAL_EXCLUDE_ZONE", "").strip()

    if not value:
        return None

    points = []

    for point in value.split(";"):
        x, y = point.split(",")
        points.append(
            [int(x), int(y)]
        )

    if len(points) < 3:
        raise ValueError(
            "EVAL_EXCLUDE_ZONE must contain at least 3 points"
        )

    return np.array(
        points,
        dtype=np.int32,
    )


def is_in_exclude_zone(
    box,
    exclude_zone,
) -> bool:
    """Return True if the centre of a detection is inside the exclusion zone."""

    if exclude_zone is None:
        return False

    center_x = box.x + box.width / 2
    center_y = box.y + box.height / 2

    return (
        cv2.pointPolygonTest(
            exclude_zone,
            (float(center_x), float(center_y)),
            False,
        )
        >= 0
    )


@app.function_name(name="ProcessFile")
@app.blob_trigger(
    arg_name="archivo",
    path="documents/in/{name}",
    connection="DocumentsStorage",
    source="EventGrid",
)
@app.blob_output(
    arg_name="salida",
    path="processed/{name}",
    connection="DocumentsStorage",
)
@app.blob_output(
    arg_name="resultado",
    path="results/{name}.json",
    connection="DocumentsStorage",
)
def procesar_archivo(
    archivo: func.InputStream,
    salida: func.Out[bytes],
    resultado: func.Out[str],
) -> None:
    """Detect people with Azure AI Vision and generate annotated output."""

    function_start = time.perf_counter()

    logging.info(
        "File received: %s | Size: %s bytes",
        archivo.name,
        archivo.length,
    )

    image_data = archivo.read()

    # --- Azure AI Vision inference ---
    vision_start = time.perf_counter()

    result = get_vision_client().analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.PEOPLE],
        language="en",
    )

    vision_ms = (
        time.perf_counter() - vision_start
    ) * 1000

    # Decode JPEG for annotation.
    image_array = np.frombuffer(
        image_data,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise ValueError("Could not decode input JPEG")

    image_height, image_width = frame.shape[:2]

    camera_id = get_camera_id(archivo.name)
    exclude_zone = (
        get_exclude_zone()
        if camera_id == "grazalema"
        else None
    )

    person_count = 0
    detected_count = 0
    excluded_count = 0

    if result.people is not None:
        for person in result.people.list:
            if person.confidence < VISION_CONFIDENCE:
                continue

            detected_count += 1

            # Bounding box provided directly by Azure AI Vision.
            box = person.bounding_box

            x1 = max(
                0,
                min(image_width - 1, int(box.x)),
            )
            y1 = max(
                0,
                min(image_height - 1, int(box.y)),
            )
            x2 = max(
                0,
                min(
                    image_width - 1,
                    int(box.x + box.width),
                ),
            )
            y2 = max(
                0,
                min(
                    image_height - 1,
                    int(box.y + box.height),
                ),
            )

            excluded = is_in_exclude_zone(
                box,
                exclude_zone,
            )

            if excluded:
                excluded_count += 1
                color = (0, 165, 255)
                label = f"Excluded {person.confidence:.2f}"
            else:
                person_count += 1
                color = (0, 0, 255)
                label = f"Person {person.confidence:.2f}"

            # Draw every detected person.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    # Draw the exclusion polygon so it is visible in the processed image.
    if exclude_zone is not None:
        cv2.polylines(
            frame,
            [exclude_zone],
            True,
            (0, 165, 255),
            2,
        )

    logging.info(
        (
            "Detected %d people | "
            "Evaluated: %d | Excluded: %d"
        ),
        detected_count,
        person_count,
        excluded_count,
    )

    # Encode processed image.
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise ValueError("Could not encode processed image")

    function_ms = (
        time.perf_counter() - function_start
    ) * 1000

    result_data = {
        "camera_id": camera_id,
        "filename": archivo.name,
        "architecture": "serverless",
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(timespec="milliseconds"),
        "input_bytes": len(image_data),
        "n_people": person_count,
        "n_people_detected": detected_count,
        "n_people_excluded": excluded_count,
        "confidence_threshold": VISION_CONFIDENCE,
        "vision_ms": round(vision_ms, 2),
        "processing_ms": round(function_ms, 2),
    }

    resultado.set(
        json.dumps(
            result_data,
            ensure_ascii=False,
        )
    )

    salida.set(encoded.tobytes())

    logging.info(
        (
            "Processed frame successfully | "
            "People: %d | Excluded: %d | "
            "vision_ms=%.1f | processing_ms=%.1f"
        ),
        person_count,
        excluded_count,
        vision_ms,
        function_ms,
    )