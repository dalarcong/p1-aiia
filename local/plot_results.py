"""
Plot counts over time from the CSV (task 1.3).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot person count over time")
    parser.add_argument("--csv", default="results/detections.csv", help="Input CSV path")
    parser.add_argument(
        "--output",
        default="results/local_people_timeline.png",
        help="Output PNG path"
    )
    parser.add_argument("--camera-id", default=None, help="Optional camera_id filter")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    timestamps: list[datetime] = []
    counts: list[int] = []

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("architecture") != "local":
                continue
            if args.camera_id is not None and row.get("camera_id") != args.camera_id:
                continue

            timestamp = datetime.fromisoformat(row["timestamp"])
            # Display in the machine's local timezone while preserving UTC in CSV
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone()
            timestamps.append(timestamp)
            counts.append(int(row["n_people"]))

    if not timestamps:
        raise SystemExit("No observations found in the CSV")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(timestamps, counts, marker="o", linewidth=1.5)
    ax.set_title("People detected per frame")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of people")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Graph written to {output_path}")


if __name__ == "__main__":
    main()