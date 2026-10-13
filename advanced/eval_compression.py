"""
Minimum resolution and JPEG compression (advanced task 3).

Re-encodes the reference frames in calibration/frames at several widths
and JPEG qualities, counts people with the production YOLO configuration
(.env) and compares the counts with the manual ground truth. The goal is
to find how much each frame can be reduced before sending it, without
losing detection effectiveness.

Everything runs locally: with a fixed YOLO_IMGSZ the size of the frame
sent does not change the inference, only the bandwidth and the detail
available to the model.

Run from the project root:
    python -m advanced.eval_compression
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from common.count_people import count_people
from common.image import decode_jpeg, encode_jpeg
from common.zones import is_excluded, load_exclude_zone

FRAMES_DIR = Path("calibration/frames")
GROUND_TRUTH_CSV = Path("calibration/ground_truth.csv")
RESULTS_DIR = Path("docs/results/adv_compression")
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
PLOT_PNG = RESULTS_DIR / "compression.png"

# 1920 is the original width of the frames (no resizing)
WIDTHS = [1920, 1280, 960, 720, 640, 480]
# 100 shows the effect of the resolution alone, almost without compression
QUALITIES = [100, 90, 75, 60, 45, 30]

FIELDS = ["width", "quality", "kib_mean", "mae", "bias", "exact", "time_ms"]


def _resize(frame: np.ndarray, width: int) -> np.ndarray:
    """
    Resize a frame to the given width, keeping its aspect ratio.

    INTER_AREA is the recommended OpenCV interpolation for shrinking:
    it averages neighbouring pixels instead of skipping them.
    Frames are never enlarged.
    """
    height, original_width = frame.shape[:2]
    if width >= original_width:
        return frame

    new_height = round(height * width / original_width)
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)


def _count_outside_zone(boxes: list[dict], scale: float, zone) -> int:
    """
    Count the boxes whose feet are outside the excluded zone.

    The zone is defined in pixels of the original 1080p frame, so the boxes
    detected on a smaller frame are scaled back to those coordinates first.
    """
    return sum(
        not is_excluded(
            box["x1"] * scale,
            box["y1"] * scale,
            box["x2"] * scale,
            box["y2"] * scale,
            zone,
        )
        for box in boxes
    )


def _evaluate(
    frames: list[tuple[np.ndarray, int]],
    width: int,
    quality: int,
    zone,
) -> dict:
    """Re-encode every frame with one width/quality and summarise the errors."""
    sizes = []
    errors = []
    times = []

    for frame, n_real in frames:
        small = _resize(frame, width)
        jpeg = encode_jpeg(small, quality)
        # len(jpeg) is exactly what would be sent over the network
        sizes.append(len(jpeg))

        start = time.perf_counter()
        _, boxes = count_people(jpeg)
        times.append((time.perf_counter() - start) * 1000)

        scale = frame.shape[1] / small.shape[1]
        errors.append(_count_outside_zone(boxes, scale, zone) - n_real)

    return {
        "width": width,
        "quality": quality,
        "kib_mean": round(statistics.mean(sizes) / 1024, 1),
        "mae": round(statistics.mean(abs(e) for e in errors), 2),
        "bias": round(statistics.mean(errors), 2),
        "exact": sum(e == 0 for e in errors),
        "time_ms": round(statistics.mean(times), 1),
    }


def _plot(rows: list[dict]) -> None:
    """MAE and frame size against the width, one line per JPEG quality."""
    fig, (ax_mae, ax_size) = plt.subplots(1, 2, figsize=(11, 4))

    for quality in QUALITIES:
        points = [row for row in rows if row["quality"] == quality]
        widths = [row["width"] for row in points]
        ax_mae.plot(widths, [row["mae"] for row in points], marker="o", label=f"q{quality}")
        ax_size.plot(widths, [row["kib_mean"] for row in points], marker="o", label=f"q{quality}")

    ax_mae.set(xlabel="Width sent (px)", ylabel="MAE (people/frame)", title="Effectiveness")
    ax_size.set(xlabel="Width sent (px)", ylabel="KiB per frame", title="Frame size")
    for ax in (ax_mae, ax_size):
        ax.set_xticks(WIDTHS)
        ax.grid(alpha=0.3)
        ax.legend(title="JPEG quality")

    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)


def main() -> None:
    # Before reading the zone: count_people and load_exclude_zone use .env
    load_dotenv()
    zone = load_exclude_zone()
    if zone is None:
        raise SystemExit("EVAL_EXCLUDE_ZONE is not set: the ground truth excludes the terrace")

    with GROUND_TRUTH_CSV.open(newline="", encoding="utf-8") as handle:
        truth = {row["frame"]: int(row["n_real"]) for row in csv.DictReader(handle)}

    # Decoded once and kept in memory (30 frames at 1080p, about 190 MB)
    frames = [
        (decode_jpeg((FRAMES_DIR / name).read_bytes()), n_real)
        for name, n_real in sorted(truth.items())
    ]

    # The first inferences are much slower (PyTorch setup): kept out of time_ms.
    # Even so, the first combination comes out ~30 % slower; time_ms is only
    # indicative here (the variable of interest is the frame size)
    for frame, _ in frames[:3]:
        count_people(encode_jpeg(frame))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for width in WIDTHS:
        for quality in QUALITIES:
            row = _evaluate(frames, width, quality, zone)
            rows.append(row)
            print(
                f"{width:>4} px q{quality:<2}: {row['kib_mean']:>6.1f} KiB "
                f"MAE={row['mae']:.2f} bias={row['bias']:+.2f} "
                f"exact={row['exact']}/{len(frames)} {row['time_ms']:.0f} ms"
            )

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _plot(rows)
    print(f"\nResults written to {SUMMARY_CSV} and {PLOT_PNG}")


if __name__ == "__main__":
    main()
