# sampled_pipeline.py
import sys
from pathlib import Path
from datetime import datetime, timedelta

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
# START_DATE = "20160316"
START_DATE = "20260307"
STEP_DAYS = 8
SAMPLE_N = 21
RANDOM_SEED = 42

# ------------------------------------------------------------------
# EDIT THESE PATHS TO MATCH YOUR PROJECT
# ------------------------------------------------------------------

# This must be the CSV created by filter.main(date),
# i.e. the input that enrich.main(date) normally reads.
FILTERED_DAY_CSV_TEMPLATE = ROOT / "data" / "processed" / "{date}" / "filtered.csv"

# Optional: save the sampled version separately as well
SAMPLED_FILTERED_CSV_TEMPLATE = ROOT / "data" / "processed" / "{date}" / "filtered_sampled_21.csv"

# Final output after relevant_urls.main(date)
FINAL_DAY_CSV_TEMPLATE = ROOT / "data" / "processed" / "{date}" / "relevant_urls.csv"

# Final sampled-across-all-days output
COMBINED_OUTPUT = ROOT / "data" / "processed" / "sampled_training_pool.csv"

# Whether to back up the full filtered file before overwriting it
BACKUP_FULL_FILTERED = True
FILTERED_BACKUP_TEMPLATE = ROOT / "data" / "processed" / "{date}" / "filtered_full_backup.csv"

# ------------------------------------------------------------------


def generate_every_8_days(start_date: str, end_date: str | None = None) -> list[str]:
    start = datetime.strptime(start_date, DATE_FMT)
    end = datetime.today() if end_date is None else datetime.strptime(end_date, DATE_FMT)

    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime(DATE_FMT))
        cur += timedelta(days=STEP_DAYS)
    return dates


def sample_filtered_candidates(
    date: str,
    sample_n: int = SAMPLE_N,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame | None:
    """
    Reads the post-filter CSV, samples up to sample_n rows,
    saves a copy of the sample, and overwrites the original filtered CSV
    so downstream steps only process the sampled rows.
    """
    filtered_path = Path(str(FILTERED_DAY_CSV_TEMPLATE).format(date=date))
    sampled_copy_path = Path(str(SAMPLED_FILTERED_CSV_TEMPLATE).format(date=date))
    backup_path = Path(str(FILTERED_BACKUP_TEMPLATE).format(date=date))

    if not filtered_path.exists():
        print(f"    [sample] Filtered CSV not found for {date}: {filtered_path}")
        return None

    df = pd.read_csv(filtered_path)

    if len(df) == 0:
        print(f"    [sample] No filtered rows for {date}")
        return None

    n = min(sample_n, len(df))
    sampled = df.sample(n=n, random_state=random_seed).copy()
    sampled.insert(0, "sample_date", date)

    filtered_path.parent.mkdir(parents=True, exist_ok=True)

    if BACKUP_FULL_FILTERED:
        df.to_csv(backup_path, index=False)
        print(f"    [sample] Backed up full filtered file to {backup_path}")

    sampled.to_csv(sampled_copy_path, index=False)
    sampled.to_csv(filtered_path, index=False)

    print(f"    [sample] Sampled {len(sampled)} / {len(df)} filtered rows")
    print(f"    [sample] Saved sampled copy to {sampled_copy_path}")
    print(f"    [sample] Overwrote filtered CSV so downstream steps use only sample")

    return sampled


def load_final_output(date: str) -> pd.DataFrame | None:
    final_path = Path(str(FINAL_DAY_CSV_TEMPLATE).format(date=date))

    if not final_path.exists():
        print(f"    [final] Final CSV not found for {date}: {final_path}")
        return None

    df = pd.read_csv(final_path)

    if len(df) == 0:
        print(f"    [final] No final rows for {date}")
        return None

    if "sample_date" not in df.columns:
        df.insert(0, "sample_date", date)

    return df


def run_one_date(date: str) -> pd.DataFrame | None:
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

    print("\n>> Cleaning up intermediates...")
    cleanup_intermediates.cleanup_day(date)

    print("\nDONE:", date)
    return load_final_output(date)


def start_pipeline(start_date: str = START_DATE, end_date: str | None = None):
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
        combined_out = Path(COMBINED_OUTPUT)
        combined_out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_out, index=False)
        print(f"\nCombined sampled output saved to: {combined_out}")
        print(f"Total final sampled rows: {len(combined)}")

    print("\nALL STEPS COMPLETE")


if __name__ == "__main__":
    start_pipeline()