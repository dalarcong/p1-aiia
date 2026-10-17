"""
Multi-camera launcher for the serverless architecture (task 4.3).

Replays previously recorded real webcam frames and uploads them to the
Azure Blob Storage input container used by the serverless application.

Each logical camera uploads one frame every `interval` seconds,
independently of whether previous frames have already been processed.
This is important because the Azure architecture is asynchronous and
must not be artificially throttled by the client.

After the sending period ends, the launcher waits for pending results
during a configurable drain period.

Results are written to:

    results/serverless_multicam/<date>_<N>cams/requests.csv
    results/serverless_multicam/<date>_<N>cams/summary.json

Run from the project root, for example:

    python -m multicam.serverless_launcher \
        --cameras 4 \
        --duration 30 \
        --replay replay_frames \
        --drain 60
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

from multicam.launcher import load_cameras


RESULTS_DIR = Path("results/serverless_multicam")


CSV_FIELDS = [
    "camera_id",
    "source_camera",
    "source_file",
    "sequence",
    "blob_name",
    "scheduled_offset_ms",
    "submitted_at",
    "frame_bytes",
    "upload_ms",
    "completed_at",
    "cloud_e2e_ms",
    "vision_ms",
    "processing_ms",
    "n_people",
    "completed_during_step",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frames from N logical cameras "
            "against the Azure serverless application"
        )
    )

    parser.add_argument(
        "--cameras",
        type=int,
        required=True,
        help="Number of logical cameras",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Sending duration in seconds (default: 60)",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between frames from each camera (default: 5)",
    )

    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("replay_frames"),
        help=(
            "Recorded frame directory "
            "(<dir>/<camera_id>/*.jpg)"
        ),
    )

    parser.add_argument(
        "--drain",
        type=float,
        default=60.0,
        help=(
            "Maximum seconds to wait for pending results "
            "after sending stops (default: 60)"
        ),
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help=(
            "Seconds between checks for new Azure results "
            "(default: 0.5)"
        ),
    )

    return parser.parse_args()


class ReplaySource:
    """Cycle through the recorded frames of one physical camera."""

    def __init__(
        self,
        camera_id: str,
        frames_dir: Path,
    ) -> None:
        paths = sorted(frames_dir.glob("*.jpg"))

        if not paths:
            raise ValueError(
                f"No replay frames found for "
                f"'{camera_id}' in {frames_dir}"
            )

        self.camera_id = camera_id
        self._frames = itertools.cycle(paths)

    def capture(self) -> tuple[str, bytes]:
        """Return the next recorded filename and JPEG bytes."""

        path = next(self._frames)

        return path.name, path.read_bytes()


def build_logical_cameras(
    n_cameras: int,
    replay_dir: Path,
) -> list[tuple[str, ReplaySource]]:
    """
    Build N logical cameras from the available recorded cameras.

    If N is larger than the number of physical recordings, the recorded
    sources are reused under new logical camera identifiers.

    Example:

        grazalema
        mojacar
        ...
        grazalema-v2
        mojacar-v2
        ...
    """

    if n_cameras <= 0:
        raise ValueError(
            "The number of cameras must be greater than zero"
        )

    physical_cameras = load_cameras()

    available: list[str] = []

    for camera_id, _ in physical_cameras:
        frames_dir = replay_dir / camera_id

        if frames_dir.exists() and any(
            frames_dir.glob("*.jpg")
        ):
            available.append(camera_id)

    if not available:
        raise SystemExit(
            f"No recorded camera frames found in {replay_dir}"
        )

    cameras: list[tuple[str, ReplaySource]] = []

    for index in range(n_cameras):
        source_id = available[index % len(available)]
        replica = index // len(available)

        if replica == 0:
            logical_id = source_id
        else:
            logical_id = (
                f"{source_id}-v{replica + 1}"
            )

        source = ReplaySource(
            camera_id=source_id,
            frames_dir=replay_dir / source_id,
        )

        cameras.append(
            (logical_id, source)
        )

    return cameras


def expected_frame_count(
    n_cameras: int,
    duration: float,
    interval: float,
) -> int:
    """
    Calculate the number of frames scheduled by the staggered launcher.

    A frame scheduled exactly at the step end is not sent.
    """

    expected = 0

    for index in range(n_cameras):
        delay = index * interval / n_cameras

        available_time = duration - delay

        if available_time <= 0:
            continue

        expected += math.ceil(
            available_time / interval
        )

    return expected


def run_serverless(
    n_cameras: int,
    duration: float,
    interval: float,
    replay_dir: Path,
    drain: float = 60.0,
    poll_interval: float = 0.5,
) -> tuple[Path, bool]:
    """
    Run one serverless multi-camera experiment.

    Returns:
        (run_directory, interrupted)
    """

    if duration <= 0:
        raise ValueError(
            "duration must be greater than zero"
        )

    if interval <= 0:
        raise ValueError(
            "interval must be greater than zero"
        )

    if drain < 0:
        raise ValueError(
            "drain must not be negative"
        )

    if poll_interval <= 0:
        raise ValueError(
            "poll_interval must be greater than zero"
        )

    load_dotenv()

    connection_string = os.environ[
        "AZURE_STORAGE_CONNECTION_STRING"
    ]

    raw_container_name = os.getenv(
        "AZURE_RAW_CONTAINER",
        "documents",
    )

    result_container_name = os.getenv(
        "AZURE_RESULTS_CONTAINER",
        "results",
    )

    service = BlobServiceClient.from_connection_string(
        connection_string
    )

    raw_container = service.get_container_client(
        raw_container_name
    )

    result_container = service.get_container_client(
        result_container_name
    )

    # Fail immediately if the configured containers do not exist
    # or cannot be accessed.
    raw_container.get_container_properties()
    result_container.get_container_properties()

    cameras = build_logical_cameras(
        n_cameras=n_cameras,
        replay_dir=replay_dir,
    )

    run_id = uuid.uuid4().hex[:8]

    run_dir = (
        RESULTS_DIR
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_"
            f"{n_cameras}cams"
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    rows: dict[str, dict] = {}

    # result blob name -> perf_counter at upload start
    pending: dict[str, float] = {}

    lock = threading.Lock()

    abort = threading.Event()
    collector_stop = threading.Event()

    # Give every thread a brief moment to start before the first
    # scheduled upload.
    step_start = time.perf_counter() + 1.0
    step_end = step_start + duration

    expected = expected_frame_count(
        n_cameras=n_cameras,
        duration=duration,
        interval=interval,
    )

    result_prefix = f"mc_{run_id}__"

    # -----------------------------------------------------------------
    # Camera producer
    # -----------------------------------------------------------------

    def camera_loop(
        logical_id: str,
        source: ReplaySource,
        start_delay: float,
    ) -> None:
        """
        Upload frames according to a fixed camera schedule.

        The next frame does not wait for Azure to finish processing the
        previous one.
        """

        sequence = 0

        scheduled = (
            step_start + start_delay
        )

        while (
            scheduled < step_end
            and not abort.is_set()
        ):
            wait = scheduled - time.perf_counter()

            if wait > 0:
                if abort.wait(wait):
                    return

            if abort.is_set():
                return

            source_file, jpeg = source.capture()

            blob_filename = (
                f"mc_{run_id}__"
                f"{logical_id}__"
                f"{sequence:05d}.jpg"
            )

            input_blob_name = (
                f"in/{blob_filename}"
            )

            result_blob_name = (
                f"{blob_filename}.json"
            )

            submitted_at = (
                datetime.now()
                .astimezone()
                .isoformat(
                    timespec="milliseconds"
                )
            )

            upload_start = time.perf_counter()

            schedule_lag_ms = (
                upload_start - scheduled
            ) * 1000.0

            row = {
                "camera_id": logical_id,
                "source_camera": source.camera_id,
                "source_file": source_file,
                "sequence": sequence,
                "blob_name": input_blob_name,
                "scheduled_offset_ms": round(
                    (
                        scheduled
                        - step_start
                    )
                    * 1000.0,
                    2,
                ),
                "submitted_at": submitted_at,
                "frame_bytes": len(jpeg),
                "upload_ms": "",
                "completed_at": "",
                "cloud_e2e_ms": "",
                "vision_ms": "",
                "processing_ms": "",
                "n_people": "",
                "completed_during_step": "",
                "status": "uploading",
            }

            # A large delay here would indicate that the local
            # uploader itself is becoming unable to maintain the
            # requested camera rate.
            if schedule_lag_ms > interval * 1000:
                print(
                    f"[CLIENT LAG] {logical_id}: "
                    f"{schedule_lag_ms:.0f} ms behind schedule"
                )

            try:
                raw_container.upload_blob(
                    name=input_blob_name,
                    data=jpeg,
                    overwrite=False,
                    content_settings=ContentSettings(
                        content_type="image/jpeg"
                    ),
                )

                upload_ms = (
                    time.perf_counter()
                    - upload_start
                ) * 1000.0

                row["upload_ms"] = round(
                    upload_ms,
                    2,
                )

                row["status"] = "pending"

                with lock:
                    rows[result_blob_name] = row

                    # cloud_e2e starts immediately before the upload,
                    # just as in the Task 3.6 benchmark.
                    pending[result_blob_name] = (
                        upload_start
                    )

            except Exception as exc:
                row["status"] = "upload_error"

                with lock:
                    rows[result_blob_name] = row

                print(
                    f"[UPLOAD ERROR] "
                    f"{logical_id}: {exc}"
                )

            sequence += 1
            scheduled += interval

    # -----------------------------------------------------------------
    # Result collector
    # -----------------------------------------------------------------

    def result_collector() -> None:
        """
        Discover result JSON blobs and match them with submitted frames.

        The collector runs independently from the camera producers.
        """

        while not collector_stop.is_set():
            with lock:
                has_pending = bool(pending)

            if not has_pending:
                time.sleep(poll_interval)
                continue

            try:
                blobs = result_container.list_blobs(
                    name_starts_with=result_prefix
                )

                for blob in blobs:
                    result_name = blob.name

                    with lock:
                        if result_name not in pending:
                            continue

                    blob_client = (
                        result_container
                        .get_blob_client(
                            result_name
                        )
                    )

                    try:
                        data = (
                            blob_client
                            .download_blob()
                            .readall()
                        )

                        # Azure Functions may expose the output blob
                        # before its contents are completely available.
                        if not data:
                            continue

                        result = json.loads(data)

                    except ResourceNotFoundError:
                        continue

                    except json.JSONDecodeError:
                        continue

                    except Exception as exc:
                        print(
                            "[COLLECTOR WARNING] "
                            f"{result_name}: {exc}"
                        )
                        continue

                    completed_perf = (
                        time.perf_counter()
                    )

                    completed_at = (
                        datetime.now()
                        .astimezone()
                        .isoformat(
                            timespec="milliseconds"
                        )
                    )

                    with lock:
                        # Recheck because another iteration could have
                        # already completed it.
                        if result_name not in pending:
                            continue

                        upload_start = pending.pop(
                            result_name
                        )

                        row = rows[result_name]

                        row["completed_at"] = (
                            completed_at
                        )

                        row["cloud_e2e_ms"] = round(
                            (
                                completed_perf
                                - upload_start
                            )
                            * 1000.0,
                            2,
                        )

                        row["vision_ms"] = (
                            result.get(
                                "vision_ms",
                                "",
                            )
                        )

                        row["processing_ms"] = (
                            result.get(
                                "processing_ms",
                                "",
                            )
                        )

                        row["n_people"] = (
                            result.get(
                                "n_people",
                                "",
                            )
                        )

                        row[
                            "completed_during_step"
                        ] = (
                            completed_perf
                            <= step_end
                        )

                        row["status"] = "ok"

            except Exception as exc:
                print(
                    f"[COLLECTOR WARNING] {exc}"
                )

            time.sleep(poll_interval)

    # -----------------------------------------------------------------
    # Start collector and camera threads
    # -----------------------------------------------------------------

    collector = threading.Thread(
        target=result_collector,
        name="result-collector",
        daemon=True,
    )

    collector.start()

    camera_threads: list[
        threading.Thread
    ] = []

    for index, (
        logical_id,
        source,
    ) in enumerate(cameras):
        # Spread all cameras uniformly across the 5 s interval.
        start_delay = (
            index * interval / n_cameras
        )

        thread = threading.Thread(
            target=camera_loop,
            args=(
                logical_id,
                source,
                start_delay,
            ),
            name=logical_id,
        )

        camera_threads.append(thread)
        thread.start()

    offered_fps = (
        n_cameras / interval
    )

    print(
        f"{n_cameras} logical cameras | "
        f"{offered_fps:.2f} offered fps | "
        f"{duration:.0f}s"
    )

    print(
        f"run_id={run_id} | "
        f"replay={replay_dir}"
    )

    print(
        f"Expected frames: {expected}"
    )

    print(
        f"Results directory: {run_dir}"
    )

    interrupted = False

    # -----------------------------------------------------------------
    # Sending period
    # -----------------------------------------------------------------

    try:
        remaining = (
            step_end
            - time.perf_counter()
        )

        if remaining > 0:
            time.sleep(remaining)

    except KeyboardInterrupt:
        interrupted = True
        abort.set()

    # Camera threads naturally stop once their next scheduled frame
    # would fall outside the experiment window.
    for thread in camera_threads:
        thread.join()

    # -----------------------------------------------------------------
    # Snapshot at the end of the load step
    # -----------------------------------------------------------------

    with lock:
        rows_snapshot = list(
            rows.values()
        )

        uploaded = sum(
            row["status"]
            != "upload_error"
            for row in rows_snapshot
        )

        upload_errors = sum(
            row["status"]
            == "upload_error"
            for row in rows_snapshot
        )

        completed_during_step = sum(
            row["status"] == "ok"
            and row[
                "completed_during_step"
            ] is True
            for row in rows_snapshot
        )

        # Frames accepted by Blob Storage but not observed as complete
        # before the step deadline.
        backlog_at_step_end = (
            uploaded
            - completed_during_step
        )

    print()
    print(
        "Sending stopped | "
        f"uploaded={uploaded}/{expected} | "
        f"completed={completed_during_step} | "
        f"backlog={backlog_at_step_end}"
    )

    # -----------------------------------------------------------------
    # Drain period
    # -----------------------------------------------------------------

    drain_deadline = (
        time.perf_counter()
        + drain
    )

    while (
        not interrupted
        and time.perf_counter()
        < drain_deadline
    ):
        with lock:
            remaining = len(pending)

        if remaining == 0:
            break

        print(
            "\rDraining Azure backlog: "
            f"{remaining} pending",
            end="",
            flush=True,
        )

        try:
            time.sleep(1.0)

        except KeyboardInterrupt:
            interrupted = True
            break

    print()

    collector_stop.set()
    collector.join()

    # -----------------------------------------------------------------
    # Mark anything still pending as timeout
    # -----------------------------------------------------------------

    with lock:
        for result_name in list(pending):
            rows[result_name][
                "status"
            ] = "timeout"

        pending.clear()

        final_rows = list(
            rows.values()
        )

    # Keep the CSV in chronological scheduled order.
    final_rows.sort(
        key=lambda row: (
            float(
                row[
                    "scheduled_offset_ms"
                ]
            ),
            row["camera_id"],
        )
    )

    # -----------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------

    csv_path = (
        run_dir / "requests.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(
            final_rows
        )

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    ok = sum(
        row["status"] == "ok"
        for row in final_rows
    )

    timeouts = sum(
        row["status"] == "timeout"
        for row in final_rows
    )

    upload_errors = sum(
        row["status"]
        == "upload_error"
        for row in final_rows
    )

    completed_during_step = sum(
        row["status"] == "ok"
        and row[
            "completed_during_step"
        ] is True
        for row in final_rows
    )

    backlog_at_step_end = (
        (len(final_rows) - upload_errors)
        - completed_during_step
    )

    summary = {
        "run_id": run_id,
        "n_cameras": n_cameras,
        "interval_s": interval,
        "duration_s": duration,
        "drain_s": drain,
        "poll_interval_s": poll_interval,
        "offered_fps": offered_fps,
        "expected_frames": expected,
        "scheduled_frames": len(
            final_rows
        ),
        "upload_errors": upload_errors,
        "completed_during_step": (
            completed_during_step
        ),
        "backlog_at_step_end": (
            backlog_at_step_end
        ),
        "completed_after_drain": ok,
        "timeouts_after_drain": timeouts,
        "interrupted": interrupted,
    }

    summary_path = (
        run_dir / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Completed during step: "
        f"{completed_during_step}/"
        f"{len(final_rows)}"
    )

    print(
        f"Completed after drain: "
        f"{ok}/{len(final_rows)}"
    )

    if upload_errors:
        print(
            f"Upload errors: "
            f"{upload_errors}"
        )

    if timeouts:
        print(
            f"Timeouts after drain: "
            f"{timeouts}"
        )

    print(
        f"Results: {csv_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    return run_dir, interrupted


def main() -> None:
    args = parse_args()

    run_serverless(
        n_cameras=args.cameras,
        duration=args.duration,
        interval=args.interval,
        replay_dir=args.replay,
        drain=args.drain,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()