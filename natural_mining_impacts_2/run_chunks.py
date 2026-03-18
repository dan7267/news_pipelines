from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


# ----------------------------
# CONFIG
# ----------------------------
BUCKET = "dan-zambia-pipeline-results"  # CHANGE
S3_PREFIX = "chunk_outputs"

BASE_RUN_DIR = Path("data/processed/pipeline_runs")
PIPELINE_SCRIPT = Path("pipeline/pipeline.py")

STEP_DAYS_DEFAULT = 14


# ----------------------------
# HELPERS
# ----------------------------
def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def chunk_ranges(start_d: date, end_d: date, step_days: int):
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


def s3_exists(s3_uri: str) -> bool:
    result = subprocess.run(
        ["aws", "s3", "ls", s3_uri],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def upload_to_s3(local_file: Path, s3_uri: str):
    subprocess.run(
        ["aws", "s3", "cp", str(local_file), s3_uri],
        check=True,
    )


def run_pipeline(start_d: date, end_d: date, run_dir: Path):
    subprocess.run(
        [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--start",
            start_d.isoformat(),
            "--end",
            end_d.isoformat(),
            "--run-dir",
            str(run_dir),
        ],
        check=True,
    )


# ----------------------------
# MAIN
# ----------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--step-days", type=int, default=STEP_DAYS_DEFAULT)

    args = parser.parse_args()

    start_d = parse_date(args.start)
    end_d = parse_date(args.end)
    step_days = args.step_days

    print(f"Running chunks from {start_d} to {end_d}")
    print(f"Chunk size: {step_days} days")
    print(f"S3 destination: s3://{BUCKET}/{S3_PREFIX}/")

    for s, e in chunk_ranges(start_d, end_d, step_days):

        chunk_label = make_chunk_label(s, e)
        run_dir = BASE_RUN_DIR / chunk_label
        local_final = run_dir / "final.csv"
        s3_uri = make_s3_uri(chunk_label)

        print(f"\n=== {chunk_label} ===")

        # Skip if already uploaded
        if s3_exists(s3_uri):
            print("Already exists in S3 → skipping")
            continue

        try:

            print(f"Running pipeline {s} → {e}")

            run_pipeline(s, e, run_dir)

            if not local_final.exists():
                raise FileNotFoundError(f"Missing {local_final}")

            print(f"Uploading {local_final}")

            upload_to_s3(local_final, s3_uri)

            print("Success")

        except Exception as err:

            print(f"FAILED: {err}")

            # continue to next chunk instead of crashing
            continue


if __name__ == "__main__":
    main()