from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pipeline.py over a date range in 14-day chunks."
    )
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument(
        "--pipeline-script",
        default="pipeline/pipeline.py",
        help="Path to your pipeline script",
    )
    parser.add_argument(
        "--base-run-dir",
        default="data/processed/pipeline_runs",
        help="Base directory where per-chunk run folders will be created",
    )
    parser.add_argument(
        "--final-filename",
        default="final.csv",
        help="Filename whose existence marks a chunk as complete",
    )
    return parser.parse_args()


def to_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    args = parse_args()

    start = to_date(args.start)
    end = to_date(args.end)

    if end < start:
        raise ValueError("--end must be on or after --start")

    chunk_days = 14
    base_run_dir = Path(args.base_run_dir)
    log_dir = base_run_dir / "_logs"

    base_run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cur = start

    while cur <= end:
        chunk_start = cur
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)

        run_name = f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}"
        run_dir = base_run_dir / run_name
        final_csv = run_dir / args.final_filename
        log_file = log_dir / f"{run_name}.log"

        if final_csv.exists():
            print(f"[skip] {run_name} already complete")
            cur = chunk_end + timedelta(days=1)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            args.pipeline_script,
            "--start",
            chunk_start.isoformat(),
            "--end",
            chunk_end.isoformat(),
            "--run-dir",
            str(run_dir),
        ]

        print(f"[run] {run_name}")
        print("      " + " ".join(cmd))
        print(f"      log -> {log_file}")

        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

        print(f"[done] {run_name}")
        cur = chunk_end + timedelta(days=1)

    print("All chunks complete.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Chunk failed with exit code {e.returncode}")
        sys.exit(e.returncode)