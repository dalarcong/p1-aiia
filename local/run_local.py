"""
Launch the complete local pipeline.

Two independent processes are started:

    Task 1.2:
        local.capture_loop
        Webcam -> frames/

    Task 1.3:
        local.detection_loop
        frames/ -> YOLO -> results/

capture_loop.py remains unchanged.

Run from the project root:
    python -m local.run_local

Short test:
    python -m local.run_local --max-frames 3
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_RESULTS_ROOT = Path("results/local")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local capture and YOLO person detection"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after the detector has processed N frames",
    )

    return parser.parse_args()


def _create_run_directory(
    results_root: Path,
) -> Path:
    """Create an independent directory for this execution."""
    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_dir = results_root / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (run_dir / "annotated").mkdir()

    return run_dir


def _stop_process(
    process: subprocess.Popen,
) -> None:
    """Stop a child process if it is still running."""
    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> None:
    args = _arguments()

    load_dotenv()

    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    results_root = Path(
        os.getenv(
            "RESULTS_ROOT",
            str(DEFAULT_RESULTS_ROOT),
        )
    )

    run_dir = _create_run_directory(
        results_root
    )

    # Any frame written after this instant can belong to this execution.
    start_time = time.time()

    detection_command = [
        sys.executable,
        "-m",
        "local.detection_loop",
        "--run-dir",
        str(run_dir),
        "--start-time",
        str(start_time),
    ]

    if args.max_frames is not None:
        detection_command += [
            "--max-frames",
            str(args.max_frames),
        ]

    capture_command = [
        sys.executable,
        "-m",
        "local.capture_loop",
    ]

    print(f"Execution directory: {run_dir}")

    print("Starting detection process (task 1.3)...")
    detection_process = subprocess.Popen(
        detection_command
    )

    print("Starting capture process (task 1.2)...")
    capture_process = subprocess.Popen(
        capture_command
    )

    try:
        while True:
            capture_status = capture_process.poll()
            detection_status = detection_process.poll()

            # During a short test, detection_loop exits normally after
            # processing --max-frames. capture_loop is then stopped here.
            if (
                args.max_frames is not None
                and detection_status == 0
            ):
                break

            # Any unexpected child-process termination stops the pipeline.
            if (
                capture_status is not None
                or detection_status is not None
            ):
                break

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping local execution...")

    finally:
        _stop_process(capture_process)
        _stop_process(detection_process)

    print(f"Results saved in: {run_dir}")
    print("Original frames remain in: frames/")
    print("Generate the graph with:")
    print(
        f"  python -m local.plot_results "
        f"--run-dir {run_dir}"
    )


if __name__ == "__main__":
    main()