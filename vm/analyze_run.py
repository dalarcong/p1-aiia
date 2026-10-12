"""
Analysis of a VM performance test (task 2.5).

Reads the CSV files of one test, saved together in a folder:

    client_requests.csv     client: latency, capture time, size, status
    server_detections.csv   server: inference time and people per frame
    vm_monitor.csv          VM: CPU, memory, swap and network every second
    effectiveness.csv       (optional) VM count vs. manual ground truth

and writes to <folder>/analysis/ a summary table (summary.csv and a
Markdown version for the report) and the graphs of the test.

Run from the project root:
    python -m vm.analyze_run docs/results/2_5_vm [--price-per-hour 0.0094]
"""
from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
MUTED = "#898781"
GRID = "#e1e0d9"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise and plot a VM performance test")
    parser.add_argument("test_dir", type=Path, help="Folder with the CSV files of the test")
    parser.add_argument(
        "--price-per-hour",
        type=float,
        default=None,
        help="VM price in EUR/hour, to estimate the cost per frame",
    )
    parser.add_argument(
        "--skip-monitor-network",
        action="store_true",
        help="Leave out the monitor network columns (wrong before the loopback fix)",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _values(rows: list[dict], field: str) -> list[float]:
    """Numeric values of a column, skipping empty cells."""
    return [float(row[field]) for row in rows if row.get(field, "") != ""]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def _describe(name: str, unit: str, values: list[float]) -> dict:
    """One summary row: mean, median, p95 and maximum of a measurement."""
    return {
        "metric": name,
        "unit": unit,
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "p95": round(_percentile(values, 0.95), 1),
        "max": round(max(values), 1),
    }


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_latency(client: list[dict], out_dir: Path) -> None:
    """Latency over time, split into inference and the rest of the request."""
    ok = [row for row in client if row["status"] == "ok"]
    times = [datetime.fromisoformat(row["timestamp"]) for row in ok]
    latency = _values(ok, "latency_ms")
    inference = _values(ok, "inference_ms")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(
        times,
        inference,
        [total - inf for total, inf in zip(latency, inference)],
        colors=[BLUE, ORANGE],
        labels=["Inference (YOLO on the VM)", "Rest (network, Apache, web publishing)"],
        alpha=0.85,
    )
    _style(ax, "Response time per frame", "Latency (ms)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=times[0].tzinfo))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_timeline.png", dpi=150)
    plt.close(fig)


def _plot_latency_distribution(client: list[dict], out_dir: Path) -> None:
    """Distribution of each time component, so they can be compared."""
    ok = [row for row in client if row["status"] == "ok"]
    latency = _values(ok, "latency_ms")
    inference = _values(ok, "inference_ms")
    components = {
        "Inference": inference,
        "Rest of request": [total - inf for total, inf in zip(latency, inference)],
        "Latency (total)": latency,
    }

    fig, ax = plt.subplots(figsize=(7, 3.5))
    box = ax.boxplot(
        list(components.values()),
        orientation="horizontal",
        tick_labels=list(components),
        patch_artist=True,
        showfliers=True,
        flierprops={"markeredgecolor": MUTED, "markersize": 4},
        medianprops={"color": "black"},
    )
    for patch, color in zip(box["boxes"], [BLUE, ORANGE, GREEN]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_title("Distribution of the response time", loc="left", fontsize=11)
    ax.set_xlabel("Time (ms)")
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_distribution.png", dpi=150)
    plt.close(fig)


def _plot_resources(monitor: list[dict], start: datetime, end: datetime, out_dir: Path) -> None:
    """CPU and memory of the VM over time, with the test window shaded."""
    times = [datetime.fromisoformat(row["timestamp"]) for row in monitor]

    def column(field: str) -> list[float | None]:
        return [float(row[field]) if row[field] != "" else None for row in monitor]

    fig, (cpu_ax, mem_ax) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax in (cpu_ax, mem_ax):
        ax.axvspan(start, end, color=GRID, alpha=0.6, label="Test running")

    cpu_ax.plot(times, column("cpu_busy_pct"), color=BLUE, linewidth=1.5, label="Whole VM")
    cpu_ax.plot(times, column("server_cpu_pct"), color=ORANGE, linewidth=1.5, label="Server process")
    cpu_ax.plot(times, column("cpu_steal_pct"), color=GREEN, linewidth=1.5, label="Steal (CPU credits)")
    _style(cpu_ax, "CPU usage", "% of the VM")
    cpu_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)

    mem_ax.plot(times, column("mem_available_mb"), color=BLUE, linewidth=1.5, label="RAM available")
    mem_ax.plot(times, column("server_rss_mb"), color=ORANGE, linewidth=1.5, label="Server process (RSS)")
    mem_ax.plot(times, column("swap_used_mb"), color=GREEN, linewidth=1.5, label="Swap used")
    _style(mem_ax, "Memory usage", "MiB")
    mem_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    mem_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=start.tzinfo))

    fig.tight_layout()
    fig.savefig(out_dir / "vm_resources.png", dpi=150)
    plt.close(fig)


def _write_summary(rows: list[dict], out_dir: Path) -> None:
    fields = ["metric", "unit", "mean", "median", "p95", "max"]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Metric | Unit | Mean | Median | p95 | Max |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [str(row.get(field, "")) for field in fields]
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _arguments()
    test_dir = args.test_dir
    out_dir = test_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    client = _read_csv(test_dir / "client_requests.csv")
    monitor = _read_csv(test_dir / "vm_monitor.csv")

    ok = [row for row in client if row["status"] == "ok"]
    start = datetime.fromisoformat(client[0]["timestamp"])
    end = datetime.fromisoformat(client[-1]["timestamp"])
    duration_s = (end - start).total_seconds()

    # Resource figures only while the test was running, not the idle margins
    during = [
        row for row in monitor
        if start <= datetime.fromisoformat(row["timestamp"]) <= end
    ]

    latency = _values(ok, "latency_ms")
    inference = _values(ok, "inference_ms")
    frame_kb = [value / 1024 for value in _values(ok, "frame_bytes")]

    rows = [
        _describe("Latency (round trip)", "ms", latency),
        _describe("Inference (YOLO)", "ms", inference),
        _describe("Rest of request", "ms", [t - i for t, i in zip(latency, inference)]),
        _describe("Capture (webcam, not VM)", "ms", _values(ok, "capture_ms")),
        _describe("CPU busy (VM)", "%", _values(during, "cpu_busy_pct")),
        _describe("CPU steal", "%", _values(during, "cpu_steal_pct")),
        _describe("CPU server process", "%", _values(during, "server_cpu_pct")),
        _describe("RAM available", "MiB", _values(during, "mem_available_mb")),
        _describe("RAM server process", "MiB", _values(during, "server_rss_mb")),
        _describe("Swap used", "MiB", _values(during, "swap_used_mb")),
        _describe("Frame size", "KiB", frame_kb),
    ]
    if not args.skip_monitor_network:
        rows += [
            _describe("VM network in", "KiB/s", _values(during, "net_recv_kbps")),
            _describe("VM network out", "KiB/s", _values(during, "net_sent_kbps")),
        ]

    # Single-value results go in the "mean" column
    upload_kbps = sum(frame_kb) / duration_s
    rows.append({"metric": "Upload bandwidth (frames)", "unit": "KiB/s", "mean": round(upload_kbps, 1)})
    rows.append({"metric": "Frames OK / sent", "unit": "frames", "mean": f"{len(ok)}/{len(client)}"})

    if args.price_per_hour is not None:
        frames_per_hour = len(ok) / duration_s * 3600
        rows.append({
            "metric": "Cost per frame",
            "unit": "EUR",
            "mean": f"{args.price_per_hour / frames_per_hour:.7f}",
        })

    effectiveness_path = test_dir / "effectiveness.csv"
    if effectiveness_path.exists():
        errors = _values(_read_csv(effectiveness_path), "error")
        rows.append({
            "metric": "Detection error (MAE / bias)",
            "unit": "people/frame",
            "mean": f"{statistics.mean(abs(e) for e in errors):.2f} / {statistics.mean(errors):+.2f}",
        })

    _write_summary(rows, out_dir)
    _plot_latency(client, out_dir)
    _plot_latency_distribution(client, out_dir)
    _plot_resources(monitor, start, end, out_dir)

    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Summary and graphs written to {out_dir}")


if __name__ == "__main__":
    main()
