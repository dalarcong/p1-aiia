"""
Shared CSV persistance for architecture comparison results.
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV_FIELDS = [
    "timestamp",
    "camera_id",
    "architecture",
    "n_people",
    "latency_ms",
    "size_bytes",
]

def append_result(
    csv_path: str | Path,
    *,
    timestamp: str,
    camera_id: str,
    architecture: str,
    n_people: int,
    latency_ms: float,
    size_bytes: int,
) -> None:
    """
    Append one observation to the CSV, creating its header if needed.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "camera_id": camera_id,
                "architecture": architecture,
                "n_people": n_people,
                "latency_ms": f"{latency_ms:.2f}",
                "size_bytes": size_bytes,
            }
        )