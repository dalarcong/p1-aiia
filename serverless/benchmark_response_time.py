"""Measure end-to-end response time of the serverless architecture.

The benchmark captures live webcam frames, uploads them to Blob Storage,
waits for the JSON result produced by the Azure Function, and records:

- capture_ms
- vision_ms
- processing_ms
- cloud_e2e_ms
- total_e2e_ms

Run from the project root:

    python -m serverless.benchmark_response_time --frames 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import numpy as np
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

from common.capture import FrameSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark serverless response time"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of live frames to process",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Maximum seconds to wait for each Azure result",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between result checks",
    )

    parser.add_argument(
        "--output",
        default="results/serverless/response_time.csv",
        help="Output CSV",
    )

    return parser.parse_args()


def wait_for_result(
    container_client,
    blob_name: str,
    timeout: float,
    poll_interval: float,
) -> dict:
    """Wait until the Function writes a valid result JSON."""

    started = time.perf_counter()

    while time.perf_counter() - started < timeout:
        blob_client = container_client.get_blob_client(
            blob_name
        )

        try:
            data = blob_client.download_blob().readall()

            # The output blob may already exist but still be empty
            # while Azure Functions is completing the output binding.
            if not data:
                time.sleep(poll_interval)
                continue

            try:
                return json.loads(data)

            except json.JSONDecodeError:
                # Blob exists but its contents are not ready yet.
                time.sleep(poll_interval)
                continue

        except ResourceNotFoundError:
            # Result blob has not been created yet.
            time.sleep(poll_interval)

    raise TimeoutError(
        f"Timed out waiting for result: {blob_name}"
    )


def summarize(
    name: str,
    values: list[float],
) -> None:
    """Print basic descriptive statistics."""

    print(f"\n{name}")
    print(f"  samples:  {len(values)}")
    print(f"  mean:     {statistics.mean(values):.2f} ms")
    print(f"  median:   {statistics.median(values):.2f} ms")
    print(f"  p95:      {np.percentile(values, 95):.2f} ms")
    print(f"  min:      {min(values):.2f} ms")
    print(f"  max:      {max(values):.2f} ms")

    if len(values) > 1:
        print(
            f"  stdev:    {statistics.stdev(values):.2f} ms"
        )


def main() -> None:
    args = parse_args()
    load_dotenv()

    if args.frames <= 0:
        raise ValueError("--frames must be greater than zero")

    connection_string = os.environ[
        "AZURE_STORAGE_CONNECTION_STRING"
    ]

    camera_url = os.environ["CAMERA_URL"]

    camera_id = os.getenv(
        "CAMERA_ID",
        "camera-1",
    )

    raw_container_name = os.getenv(
        "AZURE_RAW_CONTAINER",
        "documents",
    )

    result_container_name = os.getenv(
        "AZURE_RESULTS_CONTAINER",
        "results",
    )

    service = BlobServiceClient.from_connection_string(
        connection_string
    )

    raw_container = service.get_container_client(
        raw_container_name
    )

    result_container = service.get_container_client(
        result_container_name
    )

    source = FrameSource(camera_url)

    run_id = uuid.uuid4().hex[:8]

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []

    print(
        f"Benchmarking {args.frames} live frames "
        f"(run_id={run_id})"
    )
    print(f"Camera: {camera_id}")
    print(
        f"Polling every {args.poll_interval:.2f}s"
    )

    for index in range(1, args.frames + 1):
        total_start = time.perf_counter()

        # --- Live webcam capture ---
        capture_start = time.perf_counter()

        try:
            jpeg = source.capture()
        except Exception as exc:
            print(
                f"[{index}/{args.frames}] "
                f"CAPTURE ERROR: {exc}"
            )

            rows.append(
                {
                    "frame": index,
                    "blob_name": "",
                    "size_bytes": "",
                    "capture_ms": "",
                    "vision_ms": "",
                    "processing_ms": "",
                    "cloud_e2e_ms": "",
                    "total_e2e_ms": "",
                    "n_people": "",
                    "status": "capture_error",
                }
            )

            continue

        capture_ms = (
            time.perf_counter() - capture_start
        ) * 1000.0

        # Unique name ensures a new BlobCreated event.
        filename = (
            f"rt_{run_id}_{index:03d}.jpg"
        )

        input_blob_name = f"in/{filename}"
        result_blob_name = f"{filename}.json"

        # --- Cloud response time starts here ---
        cloud_start = time.perf_counter()

        raw_container.upload_blob(
            name=input_blob_name,
            data=jpeg,
            overwrite=False,
            content_settings=ContentSettings(
                content_type="image/jpeg"
            ),
        )

        try:
            result = wait_for_result(
                result_container,
                result_blob_name,
                args.timeout,
                args.poll_interval,
            )

        except TimeoutError:
            cloud_e2e_ms = (
                time.perf_counter() - cloud_start
            ) * 1000.0

            total_e2e_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            print(
                f"[{index}/{args.frames}] "
                f"TIMEOUT | "
                f"capture={capture_ms:.0f} ms | "
                f"cloud={cloud_e2e_ms:.0f} ms"
            )

            rows.append(
                {
                    "frame": index,
                    "blob_name": filename,
                    "size_bytes": len(jpeg),
                    "capture_ms": round(capture_ms, 2),
                    "vision_ms": "",
                    "processing_ms": "",
                    "cloud_e2e_ms": round(
                        cloud_e2e_ms,
                        2,
                    ),
                    "total_e2e_ms": round(
                        total_e2e_ms,
                        2,
                    ),
                    "n_people": "",
                    "status": "timeout",
                }
            )

            continue

        cloud_e2e_ms = (
            time.perf_counter() - cloud_start
        ) * 1000.0

        total_e2e_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        row = {
            "frame": index,
            "blob_name": filename,
            "size_bytes": len(jpeg),
            "capture_ms": round(capture_ms, 2),
            "vision_ms": result["vision_ms"],
            "processing_ms": result["processing_ms"],
            "cloud_e2e_ms": round(
                cloud_e2e_ms,
                2,
            ),
            "total_e2e_ms": round(
                total_e2e_ms,
                2,
            ),
            "n_people": result["n_people"],
            "status": "ok",
        }

        rows.append(row)

        print(
            f"[{index}/{args.frames}] "
            f"people={result['n_people']} | "
            f"capture={capture_ms:.0f} ms | "
            f"vision={float(result['vision_ms']):.0f} ms | "
            f"processing={float(result['processing_ms']):.0f} ms | "
            f"cloud={cloud_e2e_ms:.0f} ms | "
            f"total={total_e2e_ms:.0f} ms"
        )

    fieldnames = [
        "frame",
        "blob_name",
        "size_bytes",
        "capture_ms",
        "vision_ms",
        "processing_ms",
        "cloud_e2e_ms",
        "total_e2e_ms",
        "n_people",
        "status",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    successful = [
        row
        for row in rows
        if row["status"] == "ok"
    ]

    if not successful:
        raise SystemExit(
            "No frames were processed successfully."
        )

    capture_values = [
        float(row["capture_ms"])
        for row in successful
    ]

    vision_values = [
        float(row["vision_ms"])
        for row in successful
    ]

    processing_values = [
        float(row["processing_ms"])
        for row in successful
    ]

    cloud_values = [
        float(row["cloud_e2e_ms"])
        for row in successful
    ]

    total_values = [
        float(row["total_e2e_ms"])
        for row in successful
    ]

    print("\n=== Response time ===")

    summarize(
        "Webcam capture",
        capture_values,
    )

    summarize(
        "Azure AI Vision",
        vision_values,
    )

    summarize(
        "Function processing",
        processing_values,
    )

    summarize(
        "Cloud end-to-end",
        cloud_values,
    )

    summarize(
        "Full pipeline end-to-end",
        total_values,
    )

    print()
    print(
        f"Successful frames: "
        f"{len(successful)}/{args.frames}"
    )
    print(
        f"Results written to: {output_path}"
    )


if __name__ == "__main__":
    main()