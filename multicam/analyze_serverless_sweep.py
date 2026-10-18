"""
Analysis of an Azure serverless scalability sweep (task 4.3).

Reads a sweep produced by multicam/serverless_sweep.py:

    sweep_<date>/
        sweep.csv

and the run directories referenced by sweep.csv:

    <date>_<N>cams/
        requests.csv
        summary.json

Writes to:

    <sweep_dir>/analysis/
        summary.csv
        summary.md
        latency_vs_cameras.png
        throughput_vs_cameras.png
        completion_vs_cameras.png
        backlog_vs_cameras.png
        cost_per_frame_vs_cameras.png
        cost_per_hour_vs_cameras.png

Optionally, if a VM summary is supplied, also writes comparison
plots for VM vs serverless.

Run from the project root:

    python -m multicam.analyze_serverless_sweep \
        results/serverless_multicam/sweep_20261017_171703

Optional VM comparison:

    python -m multicam.analyze_serverless_sweep \
        results/serverless_multicam/sweep_20261017_171703 \
        --vm-summary docs/results/4_2_vm_sweep/analysis/summary.csv

Optional detection effectiveness:

    python -m multicam.analyze_serverless_sweep \
        results/serverless_multicam/sweep_20261017_171703 \
        --ground-truth replay_ground_truth.csv \
        --evaluation-camera grazalema
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


# Keep the same visual style used by analyze_sweep.py for the VM.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
RED = "#d9594c"
MUTED = "#898781"
GRID = "#e1e0d9"
USD_TO_EUR = 0.8924


FIELDS = [
    "cameras",
    "offered_fps",
    "frames_expected",
    "frames_uploaded",
    "upload_errors",
    "completed_during_step",
    "completion_during_step_pct",
    "backlog_at_step_end",
    "backlog_pct",
    "completed_after_drain",
    "final_completion_pct",
    "timeouts",
    "processed_fps",
    "upload_median_ms",
    "upload_p95_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "latency_steady_median_ms",
    "latency_steady_p95_ms",
    "vision_median_ms",
    "vision_p95_ms",
    "processing_median_ms",
    "processing_p95_ms",
    "serverless_overhead_median_ms",
    "detection_samples",
    "detection_mae",
    "detection_bias",
    "detection_exact_pct",
    "detection_within1_pct",
    "cost_total_eur",
    "cost_per_frame_eur",
    "cost_per_hour_eur",
]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise and plot an Azure serverless "
            "multi-camera scalability sweep"
        )
    )

    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Folder containing sweep.csv",
    )

    parser.add_argument(
        "--warmup",
        type=float,
        default=10.0,
        help=(
            "Seconds excluded from steady-state latency "
            "(default: 10)"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output folder "
            "(default: <sweep_dir>/analysis)"
        ),
    )

    parser.add_argument(
        "--vm-summary",
        type=Path,
        default=None,
        help=(
            "Optional summary.csv produced by "
            "multicam/analyze_sweep.py for VM comparison"
        ),
    )

    # Optional ground truth.
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help=(
            "Optional CSV with filename,n_people "
            "for detection-effectiveness analysis"
        ),
    )

    parser.add_argument(
        "--evaluation-camera",
        default="grazalema",
        help=(
            "Logical camera used for ground-truth evaluation "
            "(default: grazalema)"
        ),
    )

    parser.add_argument(
        "--gt-filename-column",
        default="filename",
    )

    parser.add_argument(
        "--gt-count-column",
        default="n_people",
    )

    # Normalised retail prices.
    # These are deliberately configurable so the analysis can be
    # reproduced later with updated Azure Retail Prices API values.
    parser.add_argument(
        "--vision-price-per-1000",
        type=float,
        default=1.0,
        help=(
            "Azure AI Vision price in USD per 1000 "
            "transactions (default: 1.0)"
        ),
    )

    parser.add_argument(
        "--function-gb-s-price",
        type=float,
        default=0.000026,
        help=(
            "Azure Functions Flex on-demand USD/GB-s "
            "(default: 0.000026)"
        ),
    )

    parser.add_argument(
        "--function-exec-price-per-million",
        type=float,
        default=0.40,
        help=(
            "Azure Functions execution price per million "
            "(default: 0.40 USD)"
        ),
    )

    parser.add_argument(
        "--function-memory-gb",
        type=float,
        default=2.0,
        help=(
            "Configured Function memory in GB "
            "(default: 2)"
        ),
    )

    parser.add_argument(
        "--eventgrid-price-per-million",
        type=float,
        default=0.60,
        help=(
            "Event Grid price per million operations "
            "(default: 0.60 USD)"
        ),
    )

    parser.add_argument(
        "--storage-write-price-per-10k",
        type=float,
        default=0.054,
        help=(
            "Hot LRS write operations USD/10K "
            "(default: 0.054)"
        ),
    )

    parser.add_argument(
        "--storage-read-price-per-10k",
        type=float,
        default=0.0043,
        help=(
            "Hot LRS read operations USD/10K "
            "(default: 0.0043)"
        ),
    )

    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def _values(
    rows: list[dict],
    field: str,
) -> list[float]:
    values = []

    for row in rows:
        value = row.get(field, "")

        if value not in ("", None):
            values.append(float(value))

    return values


def _percentile(
    values: list[float],
    fraction: float,
) -> float:
    if not values:
        return float("nan")

    ordered = sorted(values)

    index = round(
        fraction * (len(ordered) - 1)
    )

    return ordered[index]


def _median(
    values: list[float],
) -> float:
    if not values:
        return float("nan")

    return statistics.median(values)


def _safe_round(
    value: float,
    digits: int = 1,
):
    if math.isnan(value):
        return ""

    return round(value, digits)


def _resolve_run_dir(
    raw_path: str,
) -> Path:
    """
    Resolve the run directory stored in sweep.csv.

    serverless_sweep.py normally stores a path relative to the
    project root.
    """

    path = Path(raw_path)

    if path.exists():
        return path

    candidate = Path.cwd() / path

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Run directory not found: {raw_path}"
    )


def _load_ground_truth(
    args: argparse.Namespace,
) -> dict[str, int]:
    if args.ground_truth is None:
        return {}

    rows = _read_csv(
        args.ground_truth
    )

    truth = {}

    for row in rows:
        filename = row[
            args.gt_filename_column
        ]

        truth[filename] = int(
            row[args.gt_count_column]
        )

    return truth


def _detection_metrics(
    rows: list[dict],
    ground_truth: dict[str, int],
    camera_id: str,
) -> dict:
    """
    Evaluate unique replay images of one labelled camera.

    Repeated occurrences of the same replay frame are deduplicated,
    so a frame does not receive more weight simply because the test
    duration caused the replay source to cycle.
    """

    if not ground_truth:
        return {
            "detection_samples": "",
            "detection_mae": "",
            "detection_bias": "",
            "detection_exact_pct": "",
            "detection_within1_pct": "",
        }

    predictions: dict[str, int] = {}

    for row in rows:
        if row.get("status") != "ok":
            continue

        if row.get("camera_id") != camera_id:
            continue

        filename = row.get(
            "source_file",
            "",
        )

        if filename not in ground_truth:
            continue

        if filename not in predictions:
            predictions[filename] = int(
                row["n_people"]
            )

    if not predictions:
        return {
            "detection_samples": 0,
            "detection_mae": "",
            "detection_bias": "",
            "detection_exact_pct": "",
            "detection_within1_pct": "",
        }

    signed_errors = []
    absolute_errors = []

    for filename, predicted in predictions.items():
        expected = ground_truth[filename]

        error = predicted - expected

        signed_errors.append(error)
        absolute_errors.append(
            abs(error)
        )

    n = len(absolute_errors)

    exact = sum(
        error == 0
        for error in absolute_errors
    )

    within_one = sum(
        error <= 1
        for error in absolute_errors
    )

    return {
        "detection_samples": n,
        "detection_mae": round(
            statistics.mean(
                absolute_errors
            ),
            3,
        ),
        "detection_bias": round(
            statistics.mean(
                signed_errors
            ),
            3,
        ),
        "detection_exact_pct": round(
            exact / n * 100,
            1,
        ),
        "detection_within1_pct": round(
            within_one / n * 100,
            1,
        ),
    }


def _estimate_cost(
    rows: list[dict],
    uploaded: int,
    offered_fps: float,
    args: argparse.Namespace,
) -> dict:
    """
    Estimate normalised retail cost.

    Free grants, Azure for Students credits and storage capacity are
    deliberately excluded.

    The Function duration is approximated from processing_ms. Flex
    Consumption billing is rounded to at least one second, then to
    100 ms increments.

    The polling performed by this benchmark is not counted as part
    of the architecture's normalised storage cost.
    """

    ok = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    if not ok:
        return {
            "cost_total_eur": "",
            "cost_per_frame_eur": "",
            "cost_per_hour_eur": "",
        }

    n_ok = len(ok)

    # One PEOPLE transaction per completed frame.
    vision_cost = (
        n_ok
        * args.vision_price_per_1000
        / 1000.0
    )

    # Functions: execution count + GB-s.
    function_exec_cost = (
        n_ok
        * args.function_exec_price_per_million
        / 1_000_000.0
    )

    function_compute_cost = 0.0

    for row in ok:
        processing_s = (
            float(row["processing_ms"])
            / 1000.0
        )

        # Minimum 1 second; thereafter 100 ms increments.
        billable_s = max(
            1.0,
            math.ceil(
                processing_s * 10
            ) / 10.0,
        )

        function_compute_cost += (
            args.function_memory_gb
            * billable_s
            * args.function_gb_s_price
        )

    # Approximate Event Grid operations:
    # one publication + one delivery per uploaded raw frame.
    eventgrid_operations = (
        uploaded * 2
    )

    eventgrid_cost = (
        eventgrid_operations
        * args.eventgrid_price_per_million
        / 1_000_000.0
    )

    # Normalised Blob operations:
    # - one raw write for every uploaded frame
    # - two output writes for every successful frame
    # - one raw read for every successful Function execution
    #
    # Benchmark polling of results/ is excluded.
    storage_writes = (
        uploaded
        + 2 * n_ok
    )

    storage_reads = n_ok

    storage_cost = (
        storage_writes
        * args.storage_write_price_per_10k
        / 10_000.0
        + storage_reads
        * args.storage_read_price_per_10k
        / 10_000.0
    )

    total = (
        vision_cost
        + function_exec_cost
        + function_compute_cost
        + eventgrid_cost
        + storage_cost
    )

    per_frame = (
        total / n_ok
    )

    # Normalised cost if this offered rate were sustained for one hour.
    per_hour = (
        per_frame
        * offered_fps
        * 3600.0
    )

    return {
        "cost_total_eur": round(
            total * USD_TO_EUR,
            6,
        ),
        "cost_per_frame_eur": round(
            per_frame * USD_TO_EUR,
            7,
        ),
        "cost_per_hour_eur": round(
            per_hour * USD_TO_EUR,
            4,
        ),
    }


def _steady_rows(
    rows: list[dict],
    warmup_s: float,
) -> list[dict]:
    """
    Remove the initial scale-up/warm-up window.

    Uses submitted_at because all clients share the same machine clock.
    """

    ok = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("submitted_at")
    ]

    if not ok:
        return []

    timestamps = [
        datetime.fromisoformat(
            row["submitted_at"]
        )
        for row in ok
    ]

    start = min(timestamps)

    steady = []

    for row in ok:
        submitted = datetime.fromisoformat(
            row["submitted_at"]
        )

        age = (
            submitted - start
        ).total_seconds()

        if age >= warmup_s:
            steady.append(row)

    return steady


def _summarise_step(
    step: dict,
    ground_truth: dict[str, int],
    args: argparse.Namespace,
) -> dict:
    n_cameras = int(
        step["n_cameras"]
    )

    run_dir = _resolve_run_dir(
        step["run_dir"]
    )

    requests = _read_csv(
        run_dir / "requests.csv"
    )

    summary = json.loads(
        (
            run_dir / "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    ok = [
        row
        for row in requests
        if row.get("status") == "ok"
    ]

    steady = _steady_rows(
        requests,
        args.warmup,
    )

    upload = _values(
        requests,
        "upload_ms",
    )

    latency = _values(
        ok,
        "cloud_e2e_ms",
    )

    steady_latency = _values(
        steady,
        "cloud_e2e_ms",
    )

    vision = _values(
        ok,
        "vision_ms",
    )

    processing = _values(
        ok,
        "processing_ms",
    )

    overhead = [
        float(row["cloud_e2e_ms"])
        - float(row["processing_ms"])
        for row in ok
        if row.get("cloud_e2e_ms", "") != ""
        and row.get("processing_ms", "") != ""
    ]

    expected = int(
        summary["expected_frames"]
    )

    scheduled = int(
        summary["scheduled_frames"]
    )

    upload_errors = int(
        summary["upload_errors"]
    )

    uploaded = (
        scheduled - upload_errors
    )

    completed_during = int(
        summary[
            "completed_during_step"
        ]
    )

    completed_after = int(
        summary[
            "completed_after_drain"
        ]
    )

    backlog = int(
        summary[
            "backlog_at_step_end"
        ]
    )

    timeouts = int(
        summary[
            "timeouts_after_drain"
        ]
    )

    duration_s = float(
        summary["duration_s"]
    )

    offered_fps = float(
        summary["offered_fps"]
    )

    row = {
        "cameras": n_cameras,
        "offered_fps": round(
            offered_fps,
            2,
        ),
        "frames_expected": expected,
        "frames_uploaded": uploaded,
        "upload_errors": upload_errors,
        "completed_during_step": completed_during,
        "completion_during_step_pct": round(
            (
                completed_during
                / expected
                * 100
            )
            if expected
            else 0,
            1,
        ),
        "backlog_at_step_end": backlog,
        "backlog_pct": round(
            (
                backlog
                / expected
                * 100
            )
            if expected
            else 0,
            1,
        ),
        "completed_after_drain": completed_after,
        "final_completion_pct": round(
            (
                completed_after
                / expected
                * 100
            )
            if expected
            else 0,
            1,
        ),
        "timeouts": timeouts,
        # Throughput that was actually completed during the load step.
        "processed_fps": round(
            completed_during
            / duration_s,
            2,
        ),
        "upload_median_ms": _safe_round(
            _median(upload)
        ),
        "upload_p95_ms": _safe_round(
            _percentile(
                upload,
                0.95,
            )
        ),
        "latency_median_ms": _safe_round(
            _median(latency)
        ),
        "latency_p95_ms": _safe_round(
            _percentile(
                latency,
                0.95,
            )
        ),
        "latency_steady_median_ms": _safe_round(
            _median(
                steady_latency
            )
        ),
        "latency_steady_p95_ms": _safe_round(
            _percentile(
                steady_latency,
                0.95,
            )
        ),
        "vision_median_ms": _safe_round(
            _median(vision)
        ),
        "vision_p95_ms": _safe_round(
            _percentile(
                vision,
                0.95,
            )
        ),
        "processing_median_ms": _safe_round(
            _median(processing)
        ),
        "processing_p95_ms": _safe_round(
            _percentile(
                processing,
                0.95,
            )
        ),
        "serverless_overhead_median_ms": _safe_round(
            _median(overhead)
        ),
    }

    row.update(
        _detection_metrics(
            requests,
            ground_truth,
            args.evaluation_camera,
        )
    )

    row.update(
        _estimate_cost(
            rows=requests,
            uploaded=uploaded,
            offered_fps=offered_fps,
            args=args,
        )
    )

    return row


def _style(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.set_title(
        title,
        loc="left",
        fontsize=11,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(
        True,
        color=GRID,
        linewidth=0.8,
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)


def _plot_latency(
    rows: list[dict],
    out_dir: Path,
) -> None:
    cameras = [
        row["cameras"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        cameras,
        [
            row["latency_median_ms"]
            / 1000
            for row in rows
        ],
        color=BLUE,
        marker="o",
        linewidth=2,
        label="Cloud E2E median",
    )

    ax.plot(
        cameras,
        [
            row["latency_p95_ms"]
            / 1000
            for row in rows
        ],
        color=BLUE,
        marker="o",
        linewidth=1.5,
        linestyle="--",
        label="Cloud E2E p95",
    )

    ax.plot(
        cameras,
        [
            row["processing_median_ms"]
            / 1000
            for row in rows
        ],
        color=ORANGE,
        marker="o",
        linewidth=2,
        label="Function processing median",
    )

    ax.axhline(
        5,
        color=MUTED,
        linewidth=1.2,
        linestyle=":",
        label="5 s acquisition period",
    )

    _style(
        ax,
        "Serverless response time vs. number of cameras",
        "Cameras (1 frame every 5 s each)",
        "Time (s)",
    )

    ax.set_xticks(cameras)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        out_dir / "latency_vs_cameras.png",
        dpi=150,
    )

    plt.close(fig)


def _plot_throughput(
    rows: list[dict],
    out_dir: Path,
) -> None:
    cameras = [
        row["cameras"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        cameras,
        [
            row["offered_fps"]
            for row in rows
        ],
        color=MUTED,
        marker="o",
        linestyle="--",
        linewidth=1.5,
        label="Offered load",
    )

    ax.plot(
        cameras,
        [
            row["processed_fps"]
            for row in rows
        ],
        color=BLUE,
        marker="o",
        linewidth=2,
        label="Completed during step",
    )

    _style(
        ax,
        "Throughput vs. number of cameras",
        "Cameras (1 frame every 5 s each)",
        "Frames per second",
    )

    ax.set_xticks(cameras)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        out_dir / "throughput_vs_cameras.png",
        dpi=150,
    )

    plt.close(fig)


def _plot_completion(
    rows: list[dict],
    out_dir: Path,
) -> None:
    cameras = [
        row["cameras"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        cameras,
        [
            row[
                "completion_during_step_pct"
            ]
            for row in rows
        ],
        color=ORANGE,
        marker="o",
        linewidth=2,
        label="Completed by step end",
    )

    ax.plot(
        cameras,
        [
            row[
                "final_completion_pct"
            ]
            for row in rows
        ],
        color=GREEN,
        marker="o",
        linewidth=2,
        label="Completed after drain",
    )

    ax.axhline(
        100,
        color=MUTED,
        linewidth=1,
        linestyle=":",
    )

    _style(
        ax,
        "Completion rate vs. number of cameras",
        "Cameras",
        "Completed frames (%)",
    )

    ax.set_xticks(cameras)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        out_dir / "completion_vs_cameras.png",
        dpi=150,
    )

    plt.close(fig)


def _plot_backlog(
    rows: list[dict],
    out_dir: Path,
) -> None:
    cameras = [
        row["cameras"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        cameras,
        [
            row["backlog_at_step_end"]
            for row in rows
        ],
        color=ORANGE,
        marker="o",
        linewidth=2,
    )

    _style(
        ax,
        "Backlog at the end of each load step",
        "Cameras",
        "Pending frames",
    )

    ax.set_xticks(cameras)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    fig.savefig(
        out_dir / "backlog_vs_cameras.png",
        dpi=150,
    )

    plt.close(fig)


def _plot_cost_per_frame(
    rows: list[dict],
    out_dir: Path,
) -> None:
    cameras = [
        row["cameras"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        cameras,
        [
            row["cost_per_frame_eur"]
            for row in rows
        ],
        color=GREEN,
        marker="o",
        linewidth=2,
    )

    _style(
        ax,
        "Normalised serverless cost per frame",
        "Cameras",
        "EUR / processed frame",
    )

    ax.set_xticks(cameras)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    fig.savefig(
        out_dir / "cost_per_frame_vs_cameras.png",
        dpi=150,
    )

    plt.close(fig)


def _plot_cost_per_hour(
    rows: list[dict],
    out_dir: Path,
) -> None:
    cameras = [
        row["cameras"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        cameras,
        [
            row["cost_per_hour_eur"]
            for row in rows
        ],
        color=GREEN,
        marker="o",
        linewidth=2,
    )

    _style(
        ax,
        "Normalised serverless cost at sustained load",
        "Cameras",
        "EUR / hour",
    )

    ax.set_xticks(cameras)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    fig.savefig(
        out_dir / "cost_per_hour_vs_cameras.png",
        dpi=150,
    )

    plt.close(fig)


def _plot_effectiveness(
    rows: list[dict],
    out_dir: Path,
) -> None:
    valid = [
        row
        for row in rows
        if row["detection_mae"] != ""
    ]

    if not valid:
        return

    cameras = [
        row["cameras"]
        for row in valid
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        cameras,
        [
            row["detection_mae"]
            for row in valid
        ],
        color=BLUE,
        marker="o",
        linewidth=2,
    )

    _style(
        ax,
        "Detection effectiveness under increasing load",
        "Cameras",
        "MAE (people / frame)",
    )

    ax.set_xticks(cameras)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    fig.savefig(
        out_dir / "effectiveness_vs_cameras.png",
        dpi=150,
    )

    plt.close(fig)


def _plot_vm_comparison(
    serverless: list[dict],
    vm_summary_path: Path,
    out_dir: Path,
) -> None:
    vm = _read_csv(
        vm_summary_path
    )

    vm_cameras = [
        int(row["cameras"])
        for row in vm
    ]

    sl_cameras = [
        row["cameras"]
        for row in serverless
    ]

    # Latency comparison.
    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        vm_cameras,
        [
            float(row["latency_median_ms"])
            / 1000
            for row in vm
        ],
        marker="o",
        linewidth=2,
        color=ORANGE,
        label="VM median",
    )

    ax.plot(
        sl_cameras,
        [
            row["latency_median_ms"]
            / 1000
            for row in serverless
        ],
        marker="o",
        linewidth=2,
        color=BLUE,
        label="Serverless median",
    )

    ax.axhline(
        5,
        color=MUTED,
        linestyle=":",
        linewidth=1.2,
        label="5 s acquisition period",
    )

    _style(
        ax,
        "VM vs. serverless response time",
        "Cameras",
        "Median latency (s)",
    )

    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        out_dir / "comparison_latency.png",
        dpi=150,
    )

    plt.close(fig)

    # Throughput comparison.
    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        vm_cameras,
        [
            float(row["processed_fps"])
            for row in vm
        ],
        marker="o",
        linewidth=2,
        color=ORANGE,
        label="VM",
    )

    ax.plot(
        sl_cameras,
        [
            row["processed_fps"]
            for row in serverless
        ],
        marker="o",
        linewidth=2,
        color=BLUE,
        label="Serverless",
    )

    _style(
        ax,
        "VM vs. serverless throughput",
        "Cameras",
        "Frames processed per second",
    )

    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        out_dir / "comparison_throughput.png",
        dpi=150,
    )

    plt.close(fig)


def _write_summary_csv(
    rows: list[dict],
    path: Path,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in FIELDS
                }
            )


def _write_summary_markdown(
    rows: list[dict],
    path: Path,
) -> None:
    lines = [
        "# Serverless scalability sweep",
        "",
        "| Cameras | Offered fps | Completed / expected | Backlog | Latency med / p95 | Vision med | Processing med | Cost/frame |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['cameras']} | "
            f"{row['offered_fps']:.2f} | "
            f"{row['completed_after_drain']}/"
            f"{row['frames_expected']} | "
            f"{row['backlog_at_step_end']} | "
            f"{row['latency_median_ms']:.0f} / "
            f"{row['latency_p95_ms']:.0f} ms | "
            f"{row['vision_median_ms']:.0f} ms | "
            f"{row['processing_median_ms']:.0f} ms | "
            f"€{row['cost_per_frame_eur']:.7f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Offered fps = cameras / 5 s`.",
            "- `Processed fps` counts frames completed before the end of the load step.",
            "- `Backlog` is the number of uploaded frames not yet completed when the sending window ended.",
            "- The final completion rate is measured after the drain period.",
            "- Cloud end-to-end latency is measured from the beginning of the blob upload until the result JSON is observed.",
            f"- Steady-state latency excludes the first {rows[0].get('_warmup_s', 10):g} s of each run.",
            "- Cost is a normalised retail estimate and excludes free grants, Azure for Students credits, storage capacity and benchmark polling.",
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _print_table(
    rows: list[dict],
) -> None:
    print()
    print(
        "Cams  Offered  Done       Backlog  "
        "Latency med/p95   Vision   Proc    Cost/frame"
    )

    print(
        "----  -------  ---------  -------  "
        "----------------  -------  ------  ----------"
    )

    for row in rows:
        print(
            f"{row['cameras']:>4}  "
            f"{row['offered_fps']:>6.2f}   "
            f"{row['completed_after_drain']:>4}/"
            f"{row['frames_expected']:<4}   "
            f"{row['backlog_at_step_end']:>5}    "
            f"{row['latency_median_ms']:>5.0f}/"
            f"{row['latency_p95_ms']:<5.0f} ms   "
            f"{row['vision_median_ms']:>5.0f}   "
            f"{row['processing_median_ms']:>5.0f}   "
            f"€{row['cost_per_frame_eur']:.7f}"
        )


def main() -> None:
    args = _arguments()

    sweep_csv = (
        args.sweep_dir
        / "sweep.csv"
    )

    if not sweep_csv.exists():
        raise SystemExit(
            f"Not found: {sweep_csv}"
        )

    out_dir = (
        args.out
        if args.out is not None
        else args.sweep_dir / "analysis"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = _read_csv(
        sweep_csv
    )

    ground_truth = _load_ground_truth(
        args
    )

    rows = []

    for step in steps:
        row = _summarise_step(
            step,
            ground_truth,
            args,
        )

        # Used only for the explanatory Markdown note.
        row["_warmup_s"] = args.warmup

        rows.append(row)

    rows.sort(
        key=lambda row: row["cameras"]
    )

    _write_summary_csv(
        rows,
        out_dir / "summary.csv",
    )

    _write_summary_markdown(
        rows,
        out_dir / "summary.md",
    )

    _plot_latency(
        rows,
        out_dir,
    )

    _plot_throughput(
        rows,
        out_dir,
    )

    _plot_completion(
        rows,
        out_dir,
    )

    _plot_backlog(
        rows,
        out_dir,
    )

    _plot_cost_per_frame(
        rows,
        out_dir,
    )

    _plot_cost_per_hour(
        rows,
        out_dir,
    )

    _plot_effectiveness(
        rows,
        out_dir,
    )

    if args.vm_summary is not None:
        _plot_vm_comparison(
            rows,
            args.vm_summary,
            out_dir,
        )

    _print_table(rows)

    print()
    print(
        f"Analysis written to: "
        f"{out_dir}"
    )


if __name__ == "__main__":
    main()