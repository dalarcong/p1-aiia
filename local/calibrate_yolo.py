"""
YOLO configuration calibration.

Compares YOLOv8 models, input sizes and confidence thresholds on a fixed
set of reference frames with a manual ground truth, to choose the
configuration with the best trade-off between counting error and cost.

Two phases:
    infer:      Runs every model and input size once on every frame
                and stores all person boxes in raw_detections.csv.
    analyze:    Applies the exclusion zone and each confidence threshold to
                the stored boxes and writes summary.csv and pareto.png

Run from the project root:
    python -m local.calibrate_yolo infer
    python -m local.calibrate_yolo analyze

Quick test:
    python -m local.calibrate_yolo infer --models yolov8n.pt --sizes 640 --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from dotenv import load_dotenv
from ultralytics import YOLO

from common.count_people import PERSON_CLASS_ID
from common.zones import is_excluded, load_exclude_zone

CALIBRATION_DIR = Path("calibration")
FRAMES_DIR = CALIBRATION_DIR / "frames"
GROUND_TRUTH_CSV = CALIBRATION_DIR / "ground_truth.csv"
RAW_CSV = CALIBRATION_DIR / "raw_detections.csv"
SUMMARY_CSV = CALIBRATION_DIR / "summary.csv"
PARETO_PNG = CALIBRATION_DIR / "pareto.png"

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
IMAGE_SIZES = [640, 960, 1280]
THRESHOLDS = [0.15, 0.2, 0.25, 0.35, 0.5]

# Every box above this confidence is stored, so any higher threshold can
# be applied later in the analyze phase without running YOLO again.
MIN_CONFIDENCE = 0.05

RAW_FIELDS = ["model", "imgsz", "frame", "time_ms", "boxes"]
SUMMARY_FIELDS = ["model", "imgsz", "conf", "mae", "bias", "time_ms"]

MODEL_COLORS = {
    "yolov8n.pt": "#2a78d6",
    "yolov8s.pt": "#eb6834",
    "yolov8m.pt": "#1baf7a",
    "yolov8l.pt": "#eda100",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate YOLO model, input size and confidence threshold"
    )
    phases = parser.add_subparsers(dest="phase", required=True)

    infer = phases.add_parser("infer", help="Run YOLO on the reference frames (slow)")
    infer.add_argument("--models", nargs="+", default=MODELS)
    infer.add_argument("--sizes", nargs="+", type=int, default=IMAGE_SIZES)
    infer.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N frames (quick tests)",
    )

    analyze = phases.add_parser("analyze", help="Compute errors and plot (fast)")
    analyze.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)

    return parser.parse_args()


def _infer(
    models: list[str],
    sizes: list[int],
    limit: int | None,
    device: str,
) -> None:
    """Run every model and size on every frame and store all person boxes."""
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))[:limit]
    if not frame_paths:
        raise SystemExit(f"No frames found in {FRAMES_DIR}")

    # Decoded once up front, so JPEG decoding is not part of the measured time.
    frames = [(path.name, cv2.imread(str(path))) for path in frame_paths]

    with RAW_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()

        for model_name in models:
            model = YOLO(model_name)

            for imgsz in sizes:
                # The first inference with a new size includes one-off setup
                # costs, so it is run once and discarded.
                model.predict(frames[0][1], imgsz=imgsz, device=device, verbose=False)

                times_ms = []
                for frame_name, image in frames:
                    start = time.perf_counter()
                    result = model.predict(
                        image,
                        classes=[PERSON_CLASS_ID],
                        conf=MIN_CONFIDENCE,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                    )[0]
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    times_ms.append(elapsed_ms)

                    boxes = [
                        [round(x1), round(y1), round(x2), round(y2), round(conf, 4)]
                        for (x1, y1, x2, y2), conf in zip(
                            result.boxes.xyxy.tolist(),
                            result.boxes.conf.tolist(),
                        )
                    ]

                    writer.writerow(
                        {
                            "model": model_name,
                            "imgsz": imgsz,
                            "frame": frame_name,
                            "time_ms": f"{elapsed_ms:.1f}",
                            "boxes": json.dumps(boxes),
                        }
                    )

                # Keep finished configurations on disk if the run is interrupted
                handle.flush()
                print(
                    f"{model_name} imgsz={imgsz} "
                    f"{statistics.mean(times_ms):.0f} ms/frame"
                )
    
    print(f"Raw detections written to {RAW_CSV}")


def _load_ground_truth() -> dict[str, int]:
    """Return the manual person count of every reference frame."""
    with GROUND_TRUTH_CSV.open(newline="", encoding="utf-8") as handle:
        return {row["frame"]: int(row["n_real"]) for row in csv.DictReader(handle)}


def _analyze(thresholds: list[float]) -> None:
    """Compute counting error and cost per configuration from stored boxes."""
    if not RAW_CSV.exists():
        raise SystemExit(f"{RAW_CSV} not found: run the 'infer' phase first")
    if min(thresholds) < MIN_CONFIDENCE:
        raise SystemExit(f"Thresholds must be >= {MIN_CONFIDENCE} (stored minimum)")

    truth = _load_ground_truth()
    zone = load_exclude_zone()
    if zone is None:
        print("[WARN] EVAL_EXCLUDE_ZONE is not set: every detection is counted")

    samples: dict[tuple[str, int], list[tuple[int, list[float], float]]] = (
        defaultdict(list)
    )

    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["frame"] not in truth:
                raise SystemExit(f"{row['frame']} is missing from {GROUND_TRUTH_CSV}")

            confidences = [
                conf
                for x1, y1, x2, y2, conf in json.loads(row["boxes"])
                if not is_excluded(x1, y1, x2, y2, zone)
            ]
            samples[(row["model"], int(row["imgsz"]))].append(
                (truth[row["frame"]], confidences, float(row["time_ms"]))
            )

    summary = []
    for (model_name, imgsz), rows in samples.items():
        time_ms = statistics.mean(elapsed for _, _, elapsed in rows)

        for threshold in thresholds:
            # Positive error: YOLO counts more people than the ground truth.
            errors = [
                sum(conf >= threshold for conf in confidences) - n_real
                for n_real, confidences, _ in rows
            ]
            summary.append(
                {
                    "model": model_name,
                    "imgsz": imgsz,
                    "conf": threshold,
                    "mae": round(statistics.mean(abs(error) for error in errors), 2),
                    "bias": round(statistics.mean(errors), 2),
                    "time_ms": round(time_ms, 1)
                }
            )

    summary.sort(key=lambda row: (row["mae"], row["time_ms"]))

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    print("Best configurations (lowest MAE):")
    for row in summary[:5]:
        print(
            f"  {row['model']:<11} imgsz={row['imgsz']:<5} conf={row['conf']:<5} "
            f"MAE={row['mae']:<5} bias={row['bias']:+.2f} {row['time_ms']:.0f} ms"
        )

    _plot(summary)
    print(f"Summary written to {SUMMARY_CSV}")
    print(f"Graph written to {PARETO_PNG}")


def _plot(summary: list[dict]) -> None:
    """
    Plot counting error against inference time, one line per model.

    Each point is one input size with its best confidence threshold, so the
    trade-off is visible without one point per threshold overlapping.
    """
    best: dict[tuple[str, int], dict] = {}
    for row in summary:
        key = (row["model"], row["imgsz"])
        if key not in best or row["mae"] < best[key]["mae"]:
            best[key] = row

    fig, ax = plt.subplots(figsize=(8,5))

    for index, (model_name, color) in enumerate(MODEL_COLORS.items()):
        points = sorted(
            (row for (model, _), row in best.items() if model == model_name),
            key=lambda row: row["imgsz"]
        )
        if not points:
            continue

        ax.plot(
            [row["time_ms"] for row in points],
            [row["mae"] for row in points],
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            # White ring keeps overlapping points of different models visible
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=model_name.removesuffix(".pt"),
        )
        # Only the input size: the threshold of each point is in summary.csv,
        # and longer labels overlap between nearby points. Labels alternate
        # below/above per model, so points of different models that land on
        # the same spot do not stack their labels.
        label_offset = (0, -16) if index % 2 == 0 else (0, 10)
        for row in points:
            ax.annotate(
                f"{row['imgsz']}",
                (row["time_ms"], row["mae"]),
                textcoords="offset points",
                xytext=label_offset,
                ha="center",
                fontsize=8,
                color="#555555",
            )

    ax.set_xscale("log")
    # A few plain ticks (50, 100, 200...) instead of powers of ten, without
    # the minor ticks that overlap on a log axis
    ax.set_xticks([30, 50, 100, 200, 500, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Inference time per frame (ms, log scale)")
    ax.set_ylabel("Mean absolute error (people per frame)")
    ax.set_title(
        "YOLOv8 calibration: counting error vs. cost\n"
        "(labels: input size; each point uses its best confidence threshold)",
        fontsize=11,
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Model", frameon=False)

    fig.tight_layout()
    fig.savefig(PARETO_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _arguments()
    load_dotenv()

    if args.phase == "infer":
        _infer(args.models, args.sizes, args.limit, os.getenv("YOLO_DEVICE", "cpu"))
    else:
        _analyze(args.thresholds)


if __name__ == "__main__":
    main()