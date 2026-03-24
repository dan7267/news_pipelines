import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
import filter
import enrich
import fix_title_description
# import relevant_urls
import relevant_urls_old as relevant_urls
import cleanup_intermediates
import webscraper
import mining_matcher


DATE_FMT = "%Y%m%d"

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PIPELINE_RUNS_DIR = PROCESSED_DIR / "pipeline_runs"
MINING_MATCHED_DIR = PROCESSED_DIR / "mining_matched_daily"


def _validate_date(d: str) -> None:
    if not re.fullmatch(r"\d{8}", d or ""):
        raise ValueError(f"Invalid date '{d}'. Expected YYYYMMDD.")
    datetime.strptime(d, DATE_FMT)


def _parse_dates(user_input: str) -> list[str]:
    """
    Accepts:
      - "YYYYMMDD"
      - "YYYYMMDD-YYYYMMDD"
      - "YYYYMMDD,YYYYMMDD,YYYYMMDD"
    Returns a sorted list of YYYYMMDD strings (duplicates removed).
    """
    s = (user_input or "").strip()

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        dates = []
        for p in parts:
            _validate_date(p)
            dates.append(p)
        return sorted(set(dates))

    if "-" in s:
        a, b = [p.strip() for p in s.split("-", 1)]
        _validate_date(a)
        _validate_date(b)

        start = datetime.strptime(a, DATE_FMT)
        end = datetime.strptime(b, DATE_FMT)

        if end < start:
            start, end = end, start

        out = []
        cur = start
        while cur <= end:
            out.append(cur.strftime(DATE_FMT))
            cur += timedelta(days=1)
        return out

    _validate_date(s)
    return [s]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _mining_matched_path(date: str) -> Path:
    year, month, day = date[:4], date[4:6], date[6:8]
    return MINING_MATCHED_DIR / year / month / day / f"{date}_mining_matched.csv"


def combine_mining_matched_outputs(dates: list[str], run_dir: Path) -> None:
    frames = []

    for date in dates:
        path = _mining_matched_path(date)

        if not path.exists():
            print(f"Warning: mining matched output not found for {date}: {path}")
            continue

        try:
            df = pd.read_csv(path, encoding="utf-8", engine="python")
            df["pipeline_run_date"] = date
            df["source_file"] = str(path)
            frames.append(df)
            print(f"Included: {path}")
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")

    if not frames:
        print("\nNo mining matched daily outputs found to combine.")
        return

    combined_df = pd.concat(frames, ignore_index=True)

    combined_path = run_dir / "combined_mining_matched.csv"
    combined_df.to_csv(combined_path, index=False, encoding="utf-8")

    print("\nCombined mining matched output saved to:")
    print(combined_path)
    print(f"Combined rows: {len(combined_df):,}")


def run_one_date(date: str) -> None:
    print(f"\n==============================")
    print(f"Processing date: {date}")
    print(f"==============================")

    print(f"\n>> Step 1: Downloading...")
    download.main(date)

    print(f"\n>> Step 2: Filtering & Deduping...")
    filter.main(date)

    print(f"\n>> Step 3: Fetching Titles & Meta Tags...")
    enrich.main(date)

    print("\n>> Step 4: Cleaning Titles & Meta Tags...")
    fix_title_description.main(date)

    print("\n>> Step 5: Relevant URLs...")
    relevant_urls.main(date)

    print("\n>> Step 6: Cleaning up intermediates...")
    cleanup_intermediates.cleanup_day(date)

    print("\n>> Step 7: Webscraper...")
    webscraper.main(date)

    print("\n>> Step 8: Mining Matcher...")
    mining_matcher.main(date)

    print("\nDONE:", date)


def start_pipeline(start_date=None, end_date=None):
    """
    Can be called in two ways:

    1) Programmatically:
        start_pipeline("20240801", "20240831")

    2) From terminal / interactive input
    """

    if start_date is not None:
        if end_date is None:
            dates = _parse_dates(start_date)
        else:
            dates = _parse_dates(f"{start_date}-{end_date}")
    else:
        user_in = input(
            "Enter date(s) to process:\n"
            "  - Single: YYYYMMDD\n"
            "  - Range:  YYYYMMDD-YYYYMMDD\n"
            "  - List:   YYYYMMDD,YYYYMMDD,...\n"
            "> "
        ).strip()

        try:
            dates = _parse_dates(user_in)
        except Exception as e:
            print(f"\nError: {e}")
            return

    print(f"\nWill process {len(dates)} date(s): {', '.join(dates)}")

    for i, d in enumerate(dates, start=1):
        print(f"\n[{i}/{len(dates)}]")
        try:
            run_one_date(d)
        except Exception as e:
            print(f"\n!!! Failed for {d}: {repr(e)}")
            continue

    # Create combined run output
    if len(dates) >= 1:
        run_name = f"{dates[0]}_{dates[-1]}"
        run_dir = PIPELINE_RUNS_DIR / run_name
        _ensure_dir(run_dir)

        print(f"\n>> Combining mining matcher outputs into run folder...")
        combine_mining_matched_outputs(dates, run_dir)

    print("\nALL STEPS COMPLETE")


if __name__ == "__main__":
    start_pipeline()