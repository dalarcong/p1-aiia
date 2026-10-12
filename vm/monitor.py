"""
VM resource monitor (task 2.5).

Samples, every MONITOR_INTERVAL seconds, the CPU, memory, swap and network
usage of the whole VM and the CPU and memory of the YOLO server process,
and writes one row per sample to results/vm_monitor/<run>/system.csv.

Run on the VM from the project root, in parallel with the server:
    python -m vm.monitor
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from dotenv import load_dotenv

SERVER_ARG = "vm.server:app"

CSV_FIELDS = [
    "timestamp",
    "cpu_busy_pct",
    "cpu_steal_pct",
    "cpu_iowait_pct",
    "mem_used_mb",
    "mem_available_mb",
    "swap_used_mb",
    "server_cpu_pct",
    "server_rss_mb",
    "net_recv_kbps",
    "net_sent_kbps",
]

MB = 1024 * 1024


def _find_server() -> psutil.Process | None:
    """Return the running YOLO server process, or None if it is not running."""
    for process in psutil.process_iter(["cmdline"]):
        if SERVER_ARG in (process.info["cmdline"] or []):
            process.cpu_percent(None)
            return process
    return None


def _external_traffic() -> tuple[int, int]:
    """
    Return the bytes received and sent by the VM, excluding the loopback.

    Apache forwards every frame to FastAPI through 127.0.0.1, so counting the
    loopback interface (lo) would add each frame to the traffic twice.
    """
    counters = psutil.net_io_counters(pernic=True)
    external = [nic for name, nic in counters.items() if name != "lo"]
    return sum(nic.bytes_recv for nic in external), sum(nic.bytes_sent for nic in external)


def main() -> None:
    load_dotenv()
    interval = float(os.getenv("MONITOR_INTERVAL", "1"))
    results_root = Path(os.getenv("VM_MONITOR_ROOT", "results/vm_monitor"))

    run_dir = results_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    csv_path = run_dir / "system.csv"

    n_cpus = psutil.cpu_count()
    server = _find_server()

    # CPU and network values are differences between consecutive samples,
    # so the first readings only set the starting point
    psutil.cpu_times_percent(None)
    recv_before, sent_before = _external_traffic()
    time_before = time.monotonic()

    print(f"Monitoring every {interval}s into {csv_path} (Ctrl+C to stop)")
    if server is None:
        print("[WARN] Server process not found yet: its columns stay empty until it starts")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        try:
            while True:
                time.sleep(max(0.0, interval - (time.monotonic() - time_before)))

                cpu = psutil.cpu_times_percent(None)
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                recv, sent = _external_traffic()
                now = time.monotonic()
                elapsed = now - time_before

                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    # Time the CPU spent doing work; steal (taken by Azure when
                    # the CPU credits run out) and iowait (waiting for the disk)
                    # are reported separately
                    "cpu_busy_pct": round(
                        cpu.user + cpu.nice + cpu.system + cpu.irq + cpu.softirq, 1
                    ),
                    "cpu_steal_pct": round(cpu.steal, 1),
                    "cpu_iowait_pct": round(cpu.iowait, 1),
                    "mem_used_mb": round((memory.total - memory.available) / MB),
                    "mem_available_mb": round(memory.available / MB),
                    "swap_used_mb": round(swap.used / MB),
                    "net_recv_kbps": round((recv - recv_before) / 1024 / elapsed, 1),
                    "net_sent_kbps": round((sent - sent_before) / 1024 / elapsed, 1),
                }

                if server is None:
                    server = _find_server()
                elif server.is_running():
                    try:
                        # Process CPU is per core (up to 200 % with 2 vCPU):
                        # divided by the number of CPUs to get % of the VM
                        row["server_cpu_pct"] = round(server.cpu_percent(None) / n_cpus, 1)
                        row["server_rss_mb"] = round(server.memory_info().rss / MB)
                    except psutil.NoSuchProcess:
                        server = None
                else:
                    server = None

                writer.writerow(row)
                handle.flush()
                recv_before, sent_before, time_before = recv, sent, now
        except KeyboardInterrupt:
            print("\nStopped")


if __name__ == "__main__":
    main()
