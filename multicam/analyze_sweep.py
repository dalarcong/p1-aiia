"""
Analysis of a VM scalability sweep (task 4.2).

Reads a sweep saved in one folder:

    sweep.csv                    steps: n_cameras, start, end
    client_requests_<N>cams.csv  client CSV of each step
    vm_monitor.csv               VM monitor during the whole sweep

and writes to <folder>/analysis/ a table with one row per number of
cameras (summary.csv and a Markdown version for the report) and the graphs
that relate the number of cameras to the performance of the VM.

Run from the project root:
    python -m multicam.analyze_sweep docs/results/4_2_vm_sweep --price-per-hour 0.0168
"""
from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
MUTED = "#898781"
GRID = "#e1e0d9"

FIELDS = [
    "cameras",
    "frames_ok",
    "frames_expected",
    "errors",
    "offered_fps",
    "processed_fps",
    "latency_median_ms",
    "latency_p95_ms",
    "inference_median_ms",
    "rest_median_ms",
    "cpu_busy_mean_pct",
    "cpu_iowait_mean_pct",
    "cpu_steal_mean_pct",
    "mem_available_min_mib",
    "swap_used_max_mib",
    "server_rss_mean_mib",
    "net_in_mean_kibps",
    "cost_per_frame_eur",
]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise and plot a VM scalability sweep")
    parser.add_argument("sweep_dir", type=Path, help="Folder with the CSV files of the sweep")
    parser.add_argument("--interval", type=float, default=5, help="Seconds between frames of each camera")
    parser.add_argument("--duration", type=float, default=300, help="Planned seconds per step")
    parser.add_argument(
        "--price-per-hour",
        type=float,
        default=None,
        help="VM price in EUR/hour, to estimate the cost per frame",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: <sweep_dir>/analysis)")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _values(rows: list[dict], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field, "") != ""]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def _summarise_step(step: dict, sweep_dir: Path, monitor: list[dict], args) -> dict:
    """One row of the table: client and VM figures of one step."""
    n = int(step["n_cameras"])
    start = datetime.fromisoformat(step["start"])
    end = datetime.fromisoformat(step["end"])
    # The planned duration, not end - start: when the VM is overloaded the
    # step ends later, because it waits for the requests still in progress
    duration_s = args.duration

    client = _read_csv(sweep_dir / f"client_requests_{n}cams.csv")
    ok = [row for row in client if row["status"] == "ok"]
    latency = _values(ok, "latency_ms")
    inference = _values(ok, "inference_ms")

    # VM samples taken while this step was running
    during = [row for row in monitor if start <= datetime.fromisoformat(row["timestamp"]) <= end]

    row = {
        "cameras": n,
        "frames_ok": len(ok),
        "frames_expected": round(n * duration_s / args.interval),
        "errors": len(client) - len(ok),
        "offered_fps": round(n / args.interval, 2),
        "processed_fps": round(len(ok) / duration_s, 2),
        "latency_median_ms": round(statistics.median(latency)),
        "latency_p95_ms": round(_percentile(latency, 0.95)),
        "inference_median_ms": round(statistics.median(inference)),
        "rest_median_ms": round(statistics.median(t - i for t, i in zip(latency, inference))),
        "cpu_busy_mean_pct": round(statistics.mean(_values(during, "cpu_busy_pct")), 1),
        "cpu_iowait_mean_pct": round(statistics.mean(_values(during, "cpu_iowait_pct")), 1),
        "cpu_steal_mean_pct": round(statistics.mean(_values(during, "cpu_steal_pct")), 1),
        "mem_available_min_mib": round(min(_values(during, "mem_available_mb"))),
        "swap_used_max_mib": round(max(_values(during, "swap_used_mb"))),
        "server_rss_mean_mib": round(statistics.mean(_values(during, "server_rss_mb"))),
        "net_in_mean_kibps": round(statistics.mean(_values(during, "net_recv_kbps")), 1),
    }
    if args.price_per_hour is not None:
        # The VM costs the same per hour whatever it processes
        row["cost_per_frame_eur"] = f"{args.price_per_hour / (row['processed_fps'] * 3600):.7f}"
    return row


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_latency(rows: list[dict], interval: float, out_dir: Path) -> None:
    cameras = [row["cameras"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cameras, [r["latency_median_ms"] / 1000 for r in rows], color=BLUE, marker="o", linewidth=2, label="Latency (median)")
    ax.plot(cameras, [r["latency_p95_ms"] / 1000 for r in rows], color=BLUE, marker="o", linewidth=1.5, linestyle="--", label="Latency (p95)")
    ax.plot(cameras, [r["inference_median_ms"] / 1000 for r in rows], color=ORANGE, marker="o", linewidth=2, label="Inference (median)")
    ax.axhline(interval, color=MUTED, linewidth=1.2, linestyle=":")
    ax.annotate(f"{interval:g} s between frames", (cameras[0], interval), textcoords="offset points", xytext=(0, 4), fontsize=8, color=MUTED)
    # Log scale: latencies go from under 1 s to over 20 s
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 30])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    _style(ax, "Response time vs. number of cameras", "Cameras (1 frame every 5 s each)", "Time (s, log scale)")
    ax.set_xticks(cameras)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_cameras.png", dpi=150)
    plt.close(fig)


def _plot_throughput(rows: list[dict], out_dir: Path) -> None:
    cameras = [row["cameras"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cameras, [r["offered_fps"] for r in rows], color=MUTED, marker="o", linewidth=1.5, linestyle="--", label="Frames sent by the cameras")
    ax.plot(cameras, [r["processed_fps"] for r in rows], color=BLUE, marker="o", linewidth=2, label="Frames processed by the VM")
    _style(ax, "Throughput vs. number of cameras", "Cameras (1 frame every 5 s each)", "Frames per second")
    ax.set_xticks(cameras)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_vs_cameras.png", dpi=150)
    plt.close(fig)


def _plot_resources(rows: list[dict], out_dir: Path) -> None:
    cameras = [row["cameras"] for row in rows]
    # Two separate panels instead of one chart with two y axes
    fig, (cpu_ax, mem_ax) = plt.subplots(1, 2, figsize=(11, 4.5))

    cpu_ax.plot(cameras, [r["cpu_busy_mean_pct"] for r in rows], color=BLUE, marker="o", linewidth=2, label="CPU busy")
    cpu_ax.plot(cameras, [r["cpu_iowait_mean_pct"] for r in rows], color=ORANGE, marker="o", linewidth=2, label="Waiting for disk (iowait)")
    cpu_ax.plot(cameras, [r["cpu_steal_mean_pct"] for r in rows], color=GREEN, marker="o", linewidth=2, label="Steal (CPU credits)")
    _style(cpu_ax, "CPU (mean per step)", "Cameras", "% of the VM")
    cpu_ax.set_ylim(bottom=0)

    mem_ax.plot(cameras, [r["mem_available_min_mib"] for r in rows], color=BLUE, marker="o", linewidth=2, label="RAM available (min)")
    mem_ax.plot(cameras, [r["server_rss_mean_mib"] for r in rows], color=ORANGE, marker="o", linewidth=2, label="Server process (mean)")
    mem_ax.plot(cameras, [r["swap_used_max_mib"] for r in rows], color=GREEN, marker="o", linewidth=2, label="Swap used (max)")
    _style(mem_ax, "Memory", "Cameras", "MiB")
    mem_ax.set_ylim(bottom=0)

    for ax in (cpu_ax, mem_ax):
        ax.set_xticks(cameras)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "resources_vs_cameras.png", dpi=150)
    plt.close(fig)


def _plot_cost(rows: list[dict], out_dir: Path) -> None:
    cameras = [row["cameras"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cameras, [float(r["cost_per_frame_eur"]) * 1e6 for r in rows], color=BLUE, marker="o", linewidth=2)
    _style(ax, "Cost per processed frame vs. number of cameras", "Cameras", "Cost per frame (EUR × 10⁻⁶)")
    ax.set_xticks(cameras)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_vs_cameras.png", dpi=150)
    plt.close(fig)


def _write_summary(rows: list[dict], out_dir: Path) -> None:
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown version for the report, with the most relevant columns
    columns = [
        ("Cameras", "cameras"),
        ("Frames OK / expected", None),
        ("Processed fps", "processed_fps"),
        ("Latency median / p95 (ms)", None),
        ("Inference (ms)", "inference_median_ms"),
        ("CPU (%)", "cpu_busy_mean_pct"),
        ("iowait (%)", "cpu_iowait_mean_pct"),
        ("RAM avail. min (MiB)", "mem_available_min_mib"),
        ("Swap max (MiB)", "swap_used_max_mib"),
        ("Net in (KiB/s)", "net_in_mean_kibps"),
    ]
    if rows and "cost_per_frame_eur" in rows[0]:
        columns.append(("Cost/frame (EUR)", "cost_per_frame_eur"))

    lines = ["| " + " | ".join(title for title, _ in columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for title, field in columns:
            if title.startswith("Frames OK"):
                cells.append(f"{row['frames_ok']} / {row['frames_expected']}")
            elif title.startswith("Latency"):
                cells.append(f"{row['latency_median_ms']} / {row['latency_p95_ms']}")
            else:
                cells.append(str(row[field]))
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _arguments()
    out_dir = args.out or args.sweep_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = _read_csv(args.sweep_dir / "sweep.csv")
    monitor = _read_csv(args.sweep_dir / "vm_monitor.csv")
    rows = [_summarise_step(step, args.sweep_dir, monitor, args) for step in steps]

    _write_summary(rows, out_dir)
    _plot_latency(rows, args.interval, out_dir)
    _plot_throughput(rows, out_dir)
    _plot_resources(rows, out_dir)
    if args.price_per_hour is not None:
        _plot_cost(rows, out_dir)

    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Summary and graphs written to {out_dir}")


if __name__ == "__main__":
    main()