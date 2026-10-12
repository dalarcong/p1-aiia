"""
Detection effectiveness of the VM (task 2.5).

Sends the reference frames in calibration/frames to the VM server, using
the same contract as the client, and compares the people it counts with
the manual ground truth (calibration/ground_truth.csv). The terrace is
excluded with EVAL_EXCLUDE_ZONE, exactly as when the ground truth was
counted. With --compare-local, the same frames are also counted with the
local YOLO configuration, to check that the VM gives the same results.

Run from the project root (the VM server must be running):
    python -m vm.eval_effectiveness [--compare-local]
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from common.zones import is_excluded, load_exclude_zone
from vm.vm_client import send_frame

FRAMES_DIR = Path("calibration/frames")
GROUND_TRUTH_CSV = Path("calibration/ground_truth.csv")
RESULTS_DIR = Path("results/vm_eval")

# Kept apart from the real camera in the server CSV and the website
CAMERA_ID = "eval"

CSV_FIELDS = [
    "frame",
    "n_real",
    "n_vm",
    "error",
    "n_local",
    "latency_ms",
    "inference_ms",
]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the VM people count with the manual ground truth"
    )
    parser.add_argument(
        "--compare-local",
        action="store_true",
        help="Also count the frames locally and compare with the VM",
    )
    return parser.parse_args()


def _count_outside_zone(boxes: list[dict], zone) -> int:
    """Count the boxes whose feet are outside the excluded zone."""
    return sum(
        not is_excluded(box["x1"], box["y1"], box["x2"], box["y2"], zone)
        for box in boxes
    )


def main() -> None:
    args = _arguments()
    load_dotenv()
    vm_url = os.environ["VM_URL"].rstrip("/")
    token = os.environ["VM_TOKEN"]

    zone = load_exclude_zone()
    if zone is None:
        raise SystemExit("EVAL_EXCLUDE_ZONE is not set: the ground truth excludes the terrace")

    with GROUND_TRUTH_CSV.open(newline="", encoding="utf-8") as handle:
        truth = {row["frame"]: int(row["n_real"]) for row in csv.DictReader(handle)}

    if args.compare_local:
        # Imported only when needed: loading YOLO locally takes a few seconds
        from common.count_people import count_people

    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    csv_path = run_dir / "effectiveness.csv"

    session = requests.Session()
    rows = []

    for frame_name, n_real in sorted(truth.items()):
        jpeg = (FRAMES_DIR / frame_name).read_bytes()
        # The frame's own capture time, recovered from its file name
        captured_at = datetime.strptime(Path(frame_name).stem, "%Y%m%d_%H%M%S").astimezone()

        sent = time.perf_counter()
        result = send_frame(session, vm_url, token, CAMERA_ID, jpeg, captured_at)
        latency_ms = (time.perf_counter() - sent) * 1000

        n_vm = _count_outside_zone(result["boxes"], zone)
        row = {
            "frame": frame_name,
            "n_real": n_real,
            "n_vm": n_vm,
            "error": n_vm - n_real,
            "latency_ms": round(latency_ms, 1),
            "inference_ms": result["inference_ms"],
        }
        if args.compare_local:
            _, local_boxes = count_people(jpeg)
            row["n_local"] = _count_outside_zone(local_boxes, zone)

        rows.append(row)
        print(f"{frame_name}: real={n_real} vm={n_vm} error={row['error']:+d}")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    errors = [row["error"] for row in rows]
    print(f"\nFrames: {len(rows)}")
    print(f"MAE: {statistics.mean(abs(e) for e in errors):.2f} people/frame")
    print(f"Bias: {statistics.mean(errors):+.2f} people/frame")
    print(f"Exact count: {sum(e == 0 for e in errors)}/{len(rows)} frames")
    print(f"Mean inference: {statistics.mean(r['inference_ms'] for r in rows):.0f} ms")
    if args.compare_local:
        different = sum(row["n_vm"] != row["n_local"] for row in rows)
        print(f"Frames where VM and local counts differ: {different}/{len(rows)}")
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
