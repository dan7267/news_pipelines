from __future__ import annotations

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import List

import pandas as pd


# Change this if the file entering your first classifier has a different name
FIRST_CLASSIFIER_INPUT_NAME = "03_zambia_events_enriched.csv"


@dataclass
class SampleResult:
    sample_date: str
    success: bool
    first_classifier_rows: int | None
    run_dir: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-03-11", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-11", help="End date YYYY-MM-DD")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of random days to sample")
    parser.add_argument(
        "--pipeline-script",
        default="pipeline/pipeline.py",
        help="Path to pipeline script",
    )
    parser.add_argument(
        "--base-run-dir",
        default="data/processed/runtime_estimation_samples",
        help="Where temporary sample runs should be written",
    )
    parser.add_argument(
        "--results-csv",
        default="data/processed/runtime_estimation_samples/sample_results.csv",
        help="Where to save the per-sample results table",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help="Keep per-day run folders instead of deleting them after counting",
    )
    return parser.parse_args()


def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def random_dates(start: date, end: date, n: int, seed: int) -> List[date]:
    rng = random.Random(seed)
    all_days = (end - start).days + 1
    if n > all_days:
        raise ValueError(f"Requested {n} samples but only {all_days} days are available.")
    offsets = rng.sample(range(all_days), n)
    return sorted(start + timedelta(days=o) for o in offsets)


def count_rows(csv_path: Path) -> int:
    # Faster than loading full dataframe for very large files
    count = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)  # skip header
        for count, _ in enumerate(f, start=1):
            pass
    return count


def percentile(values: List[float], p: float) -> float:
    if not values:
        raise ValueError("No values provided")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def main() -> None:
    args = parse_args()

    start = to_date(args.start)
    end = to_date(args.end)

    sample_days = random_dates(start, end, args.n_samples, args.seed)

    base_run_dir = Path(args.base_run_dir)
    base_run_dir.mkdir(parents=True, exist_ok=True)

    results: List[SampleResult] = []

    print(f"Sampling {len(sample_days)} random days between {start} and {end}...")

    for i, d in enumerate(sample_days, start=1):
        run_name = d.strftime("%Y%m%d")
        run_dir = base_run_dir / run_name

        cmd = [
            sys.executable,
            args.pipeline_script,
            "--start", d.isoformat(),
            "--end", d.isoformat(),
            "--run-dir", str(run_dir),
            "--stop-after-enrich",
        ]

        print(f"[{i}/{len(sample_days)}] Running sample day {d} ...")

        try:
            subprocess.run(cmd, check=True)

            first_input = run_dir / FIRST_CLASSIFIER_INPUT_NAME
            if not first_input.exists():
                raise FileNotFoundError(
                    f"Expected first-classifier input file not found: {first_input}"
                )

            n_rows = count_rows(first_input)

            results.append(
                SampleResult(
                    sample_date=d.isoformat(),
                    success=True,
                    first_classifier_rows=n_rows,
                    run_dir=str(run_dir),
                    error=None,
                )
            )

            print(f"    rows into first classifier: {n_rows:,}")

        except Exception as e:
            results.append(
                SampleResult(
                    sample_date=d.isoformat(),
                    success=False,
                    first_classifier_rows=None,
                    run_dir=str(run_dir),
                    error=str(e),
                )
            )
            print(f"    failed: {e}")

    results_df = pd.DataFrame(asdict(r) for r in results)
    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_csv, index=False)

    ok = results_df[results_df["success"] == True].copy()

    if ok.empty:
        print("\nNo successful samples, so no estimate could be computed.")
        print(f"Saved results to: {results_csv}")
        return

    counts = ok["first_classifier_rows"].astype(int).tolist()

    total_days = (end - start).days + 1
    avg_rows_per_day = mean(counts)
    med_rows_per_day = median(counts)
    p10 = percentile(counts, 0.10)
    p90 = percentile(counts, 0.90)

    estimated_total_rows = avg_rows_per_day * total_days

    print("\n--- Estimation summary ---")
    print(f"Successful samples: {len(counts)}/{len(results)}")
    print(f"Mean rows/day into first classifier:   {avg_rows_per_day:,.1f}")
    print(f"Median rows/day into first classifier: {med_rows_per_day:,.1f}")
    print(f"P10 rows/day:                           {p10:,.1f}")
    print(f"P90 rows/day:                           {p90:,.1f}")
    print(f"Total days in range:                    {total_days:,}")
    print(f"Estimated total first-classifier rows:  {estimated_total_rows:,.0f}")
    print(f"\nSaved per-sample results to: {results_csv}")


if __name__ == "__main__":
    main()