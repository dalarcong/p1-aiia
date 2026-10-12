"""CSV persistence for frames processed by the Azure VM (task 2.3)."""
from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock


VM_CSV_FIELDS = [
    "captured_at",
    "received_at",
    "camera_id",
    "n_people",
    "inference_ms",
    "frame_bytes",
]

_csv_lock = Lock()


def append_vm_result(
    csv_path: str | Path,
    *,
    captured_at: str,
    received_at: str,
    camera_id: str,
    n_people: int,
    inference_ms: float,
    frame_bytes: int,
) -> None:
    """Append one processed frame to the VM CSV.

    The lock makes appends safe when FastAPI serves requests from more than
    one thread.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _csv_lock:
        needs_header = not path.exists() or path.stat().st_size == 0

        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=VM_CSV_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "captured_at": captured_at,
                    "received_at": received_at,
                    "camera_id": camera_id,
                    "n_people": n_people,
                    "inference_ms": f"{inference_ms:.1f}",
                    "frame_bytes": frame_bytes,
                }
            )
