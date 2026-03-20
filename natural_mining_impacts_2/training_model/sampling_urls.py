# sampled_urls_only_pipeline.py
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import download
import filter

DATE_FMT = "%Y%m%d"
START_DATE = "20160311"
STEP_DAYS = 8
SAMPLE_N = 100
RANDOM_SEED = 42

BASE_INTERIM_DIR = ROOT / "data" / "interim" / "gdelt_event_context_daily"
COMBINED_OUTPUT = ROOT / "data" / "sampled_urls" / "sampled_urls_only.csv"


def generate_every_8_days(start_date: str, end_date: Optional[str] = None) -> list[str]:
    start = datetime.strptime(start_date, DATE_FMT)
    end = datetime.today() if end_date is None else datetime.strptime(end_date, DATE_FMT)

    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime(DATE_FMT))
        cur += timedelta(days=STEP_DAYS)
    return dates


def get_day_dir(date: str) -> Path:
    dt = datetime.strptime(date, DATE_FMT)
    return BASE_INTERIM_DIR / dt.strftime("%Y") / dt.strftime("%m") / dt.strftime("%d")


def find_filtered_csv_for_date(date: str):
    day_dir = get_day_dir(date)

    if not day_dir.exists():
        print(f"    [find] Day dir not found: {day_dir}")
        return None

    csvs = sorted(day_dir.glob("*.csv"))
    if not csvs:
        print(f"    [find] No CSVs found in {day_dir}")
        return None

    ranked = []
    for p in csvs:
        name = p.name.lower()
        score = 0

        if "filter" in name:
            score += 40
        if "filtered" in name:
            score += 40
        if "dedup" in name:
            score += 20

        if "sample" in name:
            score -= 50
        if "backup" in name:
            score -= 50
        if "enrich" in name:
            score -= 30
        if "title" in name or "description" in name:
            score -= 30
        if "relevant" in name:
            score -= 100
        if "final" in name:
            score -= 30

        ranked.append((score, p))

    ranked.sort(key=lambda x: (-x[0], str(x[1])))
    best_score, best_path = ranked[0]

    print(f"    [find] Using filtered candidate: {best_path} (score={best_score})")
    return best_path


def sample_urls_for_date(
    date: str,
    sample_n: int = SAMPLE_N,
    random_seed: int = RANDOM_SEED,
):
    filtered_path = find_filtered_csv_for_date(date)
    if filtered_path is None:
        return None

    df = pd.read_csv(filtered_path)

    if len(df) == 0:
        print(f"    [sample] No filtered rows for {date}")
        return None

    if "sourceurl" not in df.columns:
        raise ValueError(f"'sourceurl' column not found in {filtered_path}")

    n = min(sample_n, len(df))
    sampled = df.sample(n=n, random_state=random_seed).copy()

    # Keep only the columns you actually need
    out = pd.DataFrame({
        "sample_date": date,
        "sourceurl": sampled["sourceurl"].astype(str),
    })

    # Optional: keep a day-level file too
    day_dir = get_day_dir(date)
    sampled_out_path = day_dir / f"{date}_sampled_urls_21.csv"
    out.to_csv(sampled_out_path, index=False)

    print(f"    [sample] Sampled {len(out)} / {len(df)} rows")
    print(f"    [sample] Saved daily sampled URLs to {sampled_out_path}")

    return out


def run_one_date(date: str):
    print(f"\n==============================")
    print(f"Processing date: {date}")
    print(f"==============================")

    print("\n>> Step 1: Downloading...")
    download.main(date)

    print("\n>> Step 2: Filtering & Deduping...")
    filter.main(date)

    print("\n>> Step 3: Sampling 21 URLs...")
    sampled_urls = sample_urls_for_date(date)

    print("\nDONE:", date)
    return sampled_urls


def start_pipeline(start_date: str = START_DATE, end_date: Optional[str] = None):
    dates = generate_every_8_days(start_date, end_date)

    print(f"\nWill process {len(dates)} date(s), every {STEP_DAYS} days:")
    print(", ".join(dates[:10]) + (" ..." if len(dates) > 10 else ""))

    all_samples = []

    for i, d in enumerate(dates, start=1):
        print(f"\n[{i}/{len(dates)}]")
        try:
            sampled_df = run_one_date(d)
            if sampled_df is not None and len(sampled_df) > 0:
                all_samples.append(sampled_df)
        except Exception as e:
            print(f"\n!!! Failed for {d}: {repr(e)}")
            continue

    if all_samples:
        combined = pd.concat(all_samples, ignore_index=True)
        COMBINED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(COMBINED_OUTPUT, index=False)
        print(f"\nCombined sampled URLs saved to: {COMBINED_OUTPUT}")
        print(f"Total sampled rows: {len(combined)}")

    print("\nALL STEPS COMPLETE")


if __name__ == "__main__":
    start_pipeline()