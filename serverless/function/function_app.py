import logging
import os

import azure.functions as func
import cv2
import numpy as np

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
def procesar_archivo(
    archivo: func.InputStream,
    salida: func.Out[bytes],
) -> None:
    logging.info(
        "File received: %s | Size: %s bytes",
        archivo.name,
        archivo.length,
    )

    # The Blob Trigger already gives us the image bytes.
    image_data = archivo.read()

    # Request person detections from Azure AI Vision.
    result = get_vision_client().analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.PEOPLE],
        language="en",
    )

    # Decode the JPEG so the returned bounding boxes can be drawn.
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

    person_count = 0

    if result.people is not None:
        for person in result.people.list:
            if person.confidence < VISION_CONFIDENCE:
                continue

            person_count += 1

            # Bounding box provided directly by Azure AI Vision.
            box = person.bounding_box

            x1 = int(box.x)
            y1 = int(box.y)
            x2 = int(box.x + box.width)
            y2 = int(box.y + box.height)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Person {person.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    logging.info(
        "Detected %d people in %s",
        person_count,
        archivo.name,
    )

    # Encode and write the processed image only once.
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise ValueError("Could not encode processed image")

    salida.set(encoded.tobytes())

    logging.info(
        "Processed frame written successfully | People: %d",
        person_count,
    )