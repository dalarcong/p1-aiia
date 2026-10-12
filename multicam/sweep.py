"""
VM scalability sweep (task 4.2).

Runs the multi-camera launcher with an increasing number of cameras,
one step after another, with an idle pause between steps so that the VM
goes back to rest and the steps are clearly separated in the monitor data.
Each step writes its own results/multicam/<date>_<N>cams/requests.csv, and
the sweep lists its steps (cameras, start, end, folder) in 
results/multicam/sweep_<date>.csv for the analysis.

Run from the project root, with the VM server and vm.monitor running:
    python -m multicam.sweep --steps 1 2 4 6 8 10 12 14 16 --duration 300 --pause 60
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv

from multicam.launcher import RESULTS_DIR, _load_cameras, run_cameras

SWEEP_FIELDS = ["step", "n_cameras", "start", "end", "run_dir"]

def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the launcher with an increasing number of cameras")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        required=True,
        help="Number of cameras of each step, e.g. 1 2 4 8",
    )
    parser.add_argument("--duration", type=float, default=300, help="Seconds per step")
    parser.add_argument("--pause", type=float, default=60, help="Idle seconds between steps")
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds between frames of each camera (default: CAPTURE_INTERVAL or 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    load_dotenv()
    vm_url = os.environ["VM_URL"].rstrip("/")
    token = os.environ["VM_TOKEN"]
    interval = args.interval or float(os.getenv("CAPTURE_INTERVAL", "5"))

    # Checked before starting, not when the largest step is reached
    _load_cameras(max(args.steps))

    total_s = len(args.steps) * args.duration + (len(args.steps) - 1) * args.pause
    end_estimate = datetime.now() + timedelta(seconds=total_s)
    print(f"Sweep of {len(args.steps)} steps: {args.steps} cameras")
    print(f"About {total_s / 60:.0f} min, until ~{end_estimate:%H:%M} (Ctrl+C stops it)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_path = RESULTS_DIR / f"sweep_{datetime.now():%Y%m%d_%H%M%S}.csv"

    with sweep_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS)
        writer.writeheader()

        for step, n_cameras in enumerate(args.steps, start=1):
            print(f"\n=== Step {step}/{len(args.steps)}: {n_cameras} cameras for {args.duration:.0f} s ===")
            start = datetime.now().astimezone()
            run_dir, interrupted = run_cameras(n_cameras, args.duration, interval, vm_url, token)

            writer.writerow(
                {
                    "step": step,
                    "n_cameras": n_cameras,
                    "start": start.isoformat(timespec="seconds"),
                    "end": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "run_dir": run_dir,
                }
            )
            handle.flush()

            if interrupted:
                print("Sweep interrupted")
                break
            if step < len(args.steps):
                print(f"Pause of {args.pause:.0f} s before the next step...")
                try:
                    time.sleep(args.pause)
                except KeyboardInterrupt:
                    print("Sweep interrupted")
                    break

    print(f"\nSweep steps written to {sweep_path}")


if __name__ == "__main__":
    main()