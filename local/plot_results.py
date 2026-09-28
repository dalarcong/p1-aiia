"""
Plot person counts over time for one local execution.

By default, the most recent execution in results/local/ is used.

Run from the project root:
    python -m local.plot_results

Or select one execution explicitly:
    python -m local.plot_results \
        --run-dir results/local/<execution>
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from dotenv import load_dotenv

DEFAULT_RESULTS_ROOT = Path("results/local")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot person count over time"
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Execution directory. "
            "If omitted, the latest execution is used."
        ),
    )

    return parser.parse_args()


def _latest_run(
    results_root: Path,
) -> Path:
    """Return the newest execution containing a detections.csv file."""
    if not results_root.exists():
        raise SystemExit(
            f"Results directory not found: {results_root}"
        )

    runs = [
        path
        for path in results_root.iterdir()
        if (
            path.is_dir()
            and (path / "detections.csv").is_file()
        )
    ]

    if not runs:
        raise SystemExit(
            f"No executions containing detections.csv found "
            f"in {results_root}"
        )

    return max(
        runs,
        key=lambda path: path.name,
    )


def main() -> None:
    args = _arguments()

    load_dotenv()

    results_root = Path(
        os.getenv(
            "RESULTS_ROOT",
            str(DEFAULT_RESULTS_ROOT),
        )
    )

    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else _latest_run(results_root)
    )
    csv_path = run_dir / "detections.csv"
    output_path = run_dir / "people_timeline.png"

    timestamps: list[datetime] = []
    counts: list[int] = []

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S - %d/%m/%y"))
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Graph written to {output_path}")


if __name__ == "__main__":
    main()