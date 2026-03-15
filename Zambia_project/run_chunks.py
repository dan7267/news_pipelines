from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


# ----------------------------
# CONFIG
# ----------------------------
BUCKET = "dan-zambia-pipeline-results"  # <-- CHANGE THIS
S3_PREFIX = "chunk_outputs"
BASE_RUN_DIR = Path("data/processed/pipeline_runs")
PIPELINE_SCRIPT = Path("pipeline/pipeline.py")
PYTHON_EXECUTABLE = sys.executable  # uses the current Python/venv
STEP_DAYS = 14


# ----------------------------
# HELPERS
# ----------------------------
@dataclass
class ChunkResult:
    chunk_label: str
    start: str
    end: str
    status: str
    message: str
    local_final: str
    s3_uri: str


def daterange_chunks(start_d: date, end_d: date, step_days: int):
    cur = start_d
    while cur <= end_d:
        chunk_end = min(cur + timedelta(days=step_days - 1), end_d)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def make_chunk_label(start_d: date, end_d: date) -> str:
    return f"{start_d:%Y%m%d}_{end_d:%Y%m%d}"


def make_s3_uri(chunk_label: str) -> str:
    year = chunk_label[:4]
    return f"s3://{BUCKET}/{S3_PREFIX}/{year}/{chunk_label}/final.csv"


def s3_object_exists(s3_uri: str) -> bool:
    result = subprocess.run(
        ["aws", "s3", "ls", s3_uri],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0


def upload_to_s3(local_file: Path, s3_uri: str) -> None:
    subprocess.run(
        ["aws", "s3", "cp", str(local_file), s3_uri],
        check=True,
    )


def run_pipeline(start_d: date, end_d: date, run_dir: Path) -> None:
    cmd = [
        PYTHON_EXECUTABLE,
        str(PIPELINE_SCRIPT),
        "--start",
        start_d.isoformat(),
        "--end",
        end_d.isoformat(),
        "--run-dir",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True)


def append_summary_row(summary_csv: Path, row: ChunkResult) -> None:
    write_header = not summary_csv.exists()
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "chunk_label",
                    "start",
                    "end",
                    "status",
                    "message",
                    "local_final",
                    "s3_uri",
                ]
            )
        writer.writerow(
            [
                row.chunk_label,
                row.start,
                row.end,
                row.status,
                row.message,
                row.local_final,
                row.s3_uri,
            ]
        )


# ----------------------------
# MAIN
# ----------------------------
def main():
    # ----------------------------
    # CHANGE THESE DATES
    # ----------------------------
    overall_start = date(2021, 3, 11)
    overall_end = date(2026, 3, 11)

    summary_csv = Path("data/processed/run_chunks_summary.csv")

    print(f"Running chunks from {overall_start} to {overall_end}")
    print(f"Run dir base: {BASE_RUN_DIR}")
    print(f"S3 destination: s3://{BUCKET}/{S3_PREFIX}/")

    for start_d, end_d in daterange_chunks(overall_start, overall_end, STEP_DAYS):
        chunk_label = make_chunk_label(start_d, end_d)
        run_dir = BASE_RUN_DIR / chunk_label
        local_final = run_dir / "final.csv"
        s3_uri = make_s3_uri(chunk_label)

        print(f"\n=== {chunk_label} ===")

        # Skip if already uploaded
        if s3_object_exists(s3_uri):
            msg = "Already exists in S3, skipped."
            print(msg)
            append_summary_row(
                summary_csv,
                ChunkResult(
                    chunk_label=chunk_label,
                    start=start_d.isoformat(),
                    end=end_d.isoformat(),
                    status="skipped",
                    message=msg,
                    local_final=str(local_final),
                    s3_uri=s3_uri,
                ),
            )
            continue

        try:
            print(f"Running pipeline for {start_d} to {end_d} ...")
            run_pipeline(start_d, end_d, run_dir)

            if not local_final.exists():
                raise FileNotFoundError(f"Expected final.csv not found: {local_final}")

            print(f"Uploading {local_final} to {s3_uri} ...")
            upload_to_s3(local_final, s3_uri)

            msg = "Success"
            print(msg)
            append_summary_row(
                summary_csv,
                ChunkResult(
                    chunk_label=chunk_label,
                    start=start_d.isoformat(),
                    end=end_d.isoformat(),
                    status="success",
                    message=msg,
                    local_final=str(local_final),
                    s3_uri=s3_uri,
                ),
            )

        except Exception as e:
            msg = str(e)
            print(f"FAILED: {msg}")
            append_summary_row(
                summary_csv,
                ChunkResult(
                    chunk_label=chunk_label,
                    start=start_d.isoformat(),
                    end=end_d.isoformat(),
                    status="failed",
                    message=msg,
                    local_final=str(local_final),
                    s3_uri=s3_uri,
                ),
            )
            # continue to next chunk instead of stopping everything
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()