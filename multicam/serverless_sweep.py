"""Serverless scalability sweep for task 4.3."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

from multicam.serverless_launcher import run_serverless


SWEEP_FIELDS = [
    "step",
    "n_cameras",
    "start",
    "end",
    "run_dir",
    "interrupted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a serverless multi-camera scalability sweep"
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        required=True,
        help="Camera counts for each sweep step",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--pause",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("replay_frames"),
    )

    parser.add_argument(
        "--drain",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sweep_dir = Path(
        "results/serverless_multicam"
    ) / f"sweep_{timestamp}"

    sweep_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    sweep_csv = sweep_dir / "sweep.csv"

    total_seconds = (
        len(args.steps) * args.duration
        + max(0, len(args.steps) - 1) * args.pause
    )

    estimated_end = (
        datetime.now()
        + timedelta(seconds=total_seconds)
    )

    print(
        f"Serverless sweep: {args.steps}"
    )
    print(
        f"Estimated duration: "
        f"{total_seconds / 60:.1f} min"
    )
    print(
        f"Estimated end: "
        f"{estimated_end:%H:%M:%S}"
    )

    with sweep_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SWEEP_FIELDS,
        )
        writer.writeheader()

        for step_index, n_cameras in enumerate(
            args.steps,
            start=1,
        ):
            print()
            print(
                f"=== Step {step_index}/"
                f"{len(args.steps)}: "
                f"{n_cameras} cameras ==="
            )

            start = datetime.now().astimezone()

            run_dir, interrupted = run_serverless(
                n_cameras=n_cameras,
                duration=args.duration,
                interval=args.interval,
                replay_dir=args.replay,
                drain=args.drain,
                poll_interval=args.poll_interval,
            )

            end = datetime.now().astimezone()

            writer.writerow(
                {
                    "step": step_index,
                    "n_cameras": n_cameras,
                    "start": start.isoformat(
                        timespec="seconds"
                    ),
                    "end": end.isoformat(
                        timespec="seconds"
                    ),
                    "run_dir": str(run_dir),
                    "interrupted": interrupted,
                }
            )

            handle.flush()

            if interrupted:
                print("Sweep interrupted.")
                break

            if step_index < len(args.steps):
                print(
                    f"Pause of "
                    f"{args.pause:.0f} s..."
                )
                time.sleep(args.pause)

    print()
    print(
        f"Sweep metadata written to: "
        f"{sweep_csv}"
    )


if __name__ == "__main__":
    main()