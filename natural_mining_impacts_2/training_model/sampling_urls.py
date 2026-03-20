# sampled_pipeline.py
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
import enrich
import fix_title_description
import relevant_urls
import cleanup_intermediates


DATE_FMT = "%Y%m%d"
START_DATE = "20190905"
STEP_DAYS = 8
SAMPLE_N = 21
RANDOM_SEED = 42
BACKUP_FULL_FILTERED = True

# This is where filter.py writes
BASE_INTERIM_DIR = ROOT / "data" / "interim" / "gdelt_event_context_daily"

# This is just where you want the combined final sampled output
COMBINED_OUTPUT = ROOT / "data" / "processed" / "sampled_training_pool.csv"


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

    print(f"    [find] CSVs in {day_dir}:")
    for p in csvs:
        print(f"      - {p.name}")

    ranked = []
    for p in csvs:
        name = p.name.lower()
        score = 0

        # prefer likely post-filter files
        if "filter" in name:
            score += 40
        if "filtered" in name:
            score += 40
        if "dedup" in name:
            score += 20
        if "analyse" in name or "analyze" in name:
            score += 10

        # avoid obviously later-stage / backup/sample files
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


def sample_filtered_candidates(
    date: str,
    sample_n: int = SAMPLE_N,
    random_seed: int = RANDOM_SEED,
):
    filtered_path = find_filtered_csv_for_date(date)
    if filtered_path is None:
        return None

    sampled_copy_path = filtered_path.with_name(filtered_path.stem + "_sampled_21.csv")
    backup_path = filtered_path.with_name(filtered_path.stem + "_full_backup.csv")

    df = pd.read_csv(filtered_path)

    if len(df) == 0:
        print(f"    [sample] No filtered rows for {date}")
        return None

    n = min(sample_n, len(df))
    sampled = df.sample(n=n, random_state=random_seed).copy()

    if "sample_date" not in sampled.columns:
        sampled.insert(0, "sample_date", date)

    if BACKUP_FULL_FILTERED:
        df.to_csv(backup_path, index=False)
        print(f"    [sample] Backed up full filtered file to {backup_path}")

    sampled.to_csv(sampled_copy_path, index=False)
    sampled.to_csv(filtered_path, index=False)

    print(f"    [sample] Sampled {len(sampled)} / {len(df)} filtered rows")
    print(f"    [sample] Saved sampled copy to {sampled_copy_path}")
    print(f"    [sample] Overwrote filtered CSV so downstream steps use only sample")

    return sampled


def find_final_csv_for_date(date: str):
    day_dir = get_day_dir(date)

    if not day_dir.exists():
        print(f"    [final] Day dir not found: {day_dir}")
        return None

    csvs = sorted(day_dir.glob("*.csv"))
    if not csvs:
        print(f"    [final] No CSVs found in {day_dir}")
        return None

    ranked = []
    for p in csvs:
        name = p.name.lower()
        score = 0

        if "relevant" in name:
            score += 50
        if "final" in name:
            score += 20
        if "sample" in name:
            score -= 10
        if "backup" in name:
            score -= 20
        if "filtered" in name:
            score -= 30

        ranked.append((score, p))

    ranked.sort(key=lambda x: (-x[0], str(x[1])))
    best_score, best_path = ranked[0]

    print(f"    [final] Using final candidate: {best_path} (score={best_score})")
    return best_path


def load_final_output(date: str):
    final_path = find_final_csv_for_date(date)
    if final_path is None:
        return None

    df = pd.read_csv(final_path)

    if len(df) == 0:
        print(f"    [final] No final rows for {date}")
        return None

    if "sample_date" not in df.columns:
        df.insert(0, "sample_date", date)

    return df


def run_one_date(date: str):
    print(f"\n==============================")
    print(f"Processing date: {date}")
    print(f"==============================")

    print(f"\n>> Step 1: Downloading...")
    download.main(date)

    print(f"\n>> Step 2: Filtering & Deduping...")
    filter.main(date)

    print(f"\n>> Step 3: Sampling 21 rows BEFORE enrichment...")
    sampled_df = sample_filtered_candidates(date)
    if sampled_df is None or len(sampled_df) == 0:
        print(f"    [sample] Skipping downstream steps for {date} because no sampled rows")
        return None

    print(f"\n>> Step 4: Fetching Titles & Meta Tags for sampled rows only...")
    enrich.main(date)

    print("\n>> Step 5: Cleaning Titles & Meta Tags...")
    fix_title_description.main(date)

    print("\n>> Step 6: Relevant URLs...")
    relevant_urls.main(date)

    final_df = load_final_output(date)

    print("\n>> Cleaning up intermediates...")
    cleanup_intermediates.cleanup_day(date)

    print("\nDONE:", date)
    return final_df


def start_pipeline(start_date: str = START_DATE, end_date: Optional[str] = None):
    dates = generate_every_8_days(start_date, end_date)

    print(f"\nWill process {len(dates)} date(s), every {STEP_DAYS} days:")
    print(", ".join(dates[:10]) + (" ..." if len(dates) > 10 else ""))

    all_sampled_final = []

    for i, d in enumerate(dates, start=1):
        print(f"\n[{i}/{len(dates)}]")
        try:
            final_df = run_one_date(d)
            if final_df is not None and len(final_df) > 0:
                all_sampled_final.append(final_df)
        except Exception as e:
            print(f"\n!!! Failed for {d}: {repr(e)}")
            continue

    if all_sampled_final:
        combined = pd.concat(all_sampled_final, ignore_index=True)
        COMBINED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(COMBINED_OUTPUT, index=False)
        print(f"\nCombined sampled output saved to: {COMBINED_OUTPUT}")
        print(f"Total final sampled rows: {len(combined)}")

    print("\nALL STEPS COMPLETE")


if __name__ == "__main__":
    start_pipeline()