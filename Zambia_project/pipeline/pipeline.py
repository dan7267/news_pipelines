"""pipeline.py

End-to-end pipeline runner.

Order:
  1) download.py
  2) analyse_raw.py
  3) fetch_metadata.py
  4) first_classifier.py
  5) second_classifier.py

Interactive inputs (if not provided as CLI args):
  - start date (YYYY-MM-DD)
  - end date   (YYYY-MM-DD)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

# local stages
import download as download_stage
from analyse_raw import run_analyse_raw
from fetch_metadata import enrich_file
from first_classifier import run_filter, keep_remaining
from second_classifier import run_stage2




# ------------------ STATS HELPERS ------------------ #

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent

def build_final_dataset(
    step6_csv: Path,
    out_csv: Path,
) -> Path:
    """
    Create final.csv:
      - Keep only mining_related == True
      - Keep only in_zambia == True
      - Keep only rows with impact_evidence
      - Keep selected columns only
    """

    import pandas as pd

    if not step6_csv.exists():
        raise FileNotFoundError(f"Step 6 file not found: {step6_csv}")

    df = pd.read_csv(step6_csv)

    # ----------- Find mining-related column safely -----------
    mining_col = None
    for c in ["mining_related", "is_mining_related"]:
        if c in df.columns:
            mining_col = c
            break

    if mining_col is None:
        raise ValueError("No mining_related column found in step 6 output.")

    # ----------- Find in_zambia column safely -----------
    zambia_col = None
    for c in ["in_zambia", "is_in_zambia"]:
        if c in df.columns:
            zambia_col = c
            break

    if zambia_col is None:
        raise ValueError("No in_zambia column found in step 6 output.")

    # ----------- Apply filters -----------
    df_filtered = df[
        (df[mining_col] == True) &
        (df[zambia_col] == True)
    ].copy()

    # ----------- Require populated impact levels -----------

    impact_level_cols = [
        c for c in df_filtered.columns
        if c.lower().startswith("impact_level")
    ]

    if impact_level_cols:
        level_mask = (
            df_filtered[impact_level_cols]
            .fillna("")
            .astype(str)
            .apply(lambda x: x.str.strip() != "")
            .all(axis=1)   # all levels must be present
        )

        df_filtered = df_filtered[level_mask]

    # ----------- Build list of desired columns -----------

    impact_level_cols = [
        c for c in df_filtered.columns
        if c.lower().startswith("impact_level")
        or c.lower().startswith("impact_focus")
    ]

    desired_cols = [
        "sourceurl",
        "sqldate",
        "title",
        "scrape_status",
        "in_zambia_confidence",
        "mining_related_confidence",
        "impact_confidence",
        "impact_evidence",
        "mine_name",
        "region",
        "mineral_type",
        "mining_company",
    ]

    final_cols = [c for c in desired_cols if c in df_filtered.columns]

    for c in impact_level_cols:
        if c not in final_cols:
            final_cols.append(c)

    df_final = df_filtered[final_cols]

    if "impact_evidence" in df_final.columns:
        df_final["impact_evidence"] = df_final["impact_evidence"].map(clean_text)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_csv, index=False)

    print(f"Final filtered dataset written: {out_csv}")
    print(f"Rows in final dataset: {len(df_final)}")

    return out_csv


def _now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_bool_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Return a boolean series for col if present, else None."""
    if col not in df.columns:
        return None
    s = df[col]
    # handle bools, 0/1, strings
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    # strings
    ss = s.fillna("").astype(str).str.strip().str.lower()
    return ss.isin(["true", "t", "1", "yes", "y"])


def _count_true(df: pd.DataFrame, col: str) -> Optional[int]:
    b = _safe_bool_series(df, col)
    if b is None:
        return None
    return int(b.sum())


def _col_present(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _value_counts(df: pd.DataFrame, col: str, top_n: int = 50) -> Dict[str, int]:
    s = df[col].fillna("").astype(str).str.strip()
    s = s.replace({"": "UNKNOWN"})
    vc = s.value_counts().head(top_n)
    return {str(k): int(v) for k, v in vc.items()}


def _domain_counts(df: pd.DataFrame, url_col: str = "sourceurl", top_n: int = 50) -> Dict[str, int]:
    if url_col not in df.columns:
        return {}
    dom = (
        df[url_col]
        .fillna("")
        .astype(str)
        .map(lambda u: urlparse(u).netloc.lower().strip() if u else "")
    )
    dom = dom.replace({"": "UNKNOWN"})
    vc = dom.value_counts().head(top_n)
    return {str(k): int(v) for k, v in vc.items()}


def _error_reason_counts(df: pd.DataFrame, error_col: str, top_n: int = 50) -> Dict[str, int]:
    s = df[error_col].fillna("").astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return {}
    # normalize whitespace and truncate
    s = s.str.replace(r"\s+", " ", regex=True).str.slice(0, 220)
    vc = s.value_counts().head(top_n)
    return {str(k): int(v) for k, v in vc.items()}


@dataclass
class RunStats:
    run_id: str
    start_date: str
    end_date: str
    created_utc: str = field(default_factory=_now_utc)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def dump_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "created_utc": self.created_utc,
            **self.metrics,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_runs_csv(self, path: Path) -> None:
        """Append ONE row per run (wide format), with selected columns removed."""
        drop_cols = {
            "run_id",
            "start_date",
            "end_date",
            "created_utc",
            "post_first_classifier_reaining_rows_total",  # (as requested)
            "post_first_classifier_remaining_rows_total", # (in case correct spelling exists)
            "final_rows_total",
            "final_unique_sourceurl",
            "final_has_title_rate",
            "final_has_description_rate",
        }

        payload = {
            "run_id": self.run_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "created_utc": self.created_utc,
            **self.metrics,
        }

        # Drop unwanted keys
        payload = {k: v for k, v in payload.items() if k not in drop_cols}

        df = pd.DataFrame([payload])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df.to_csv(path, index=False, encoding="utf-8")


def write_long_table(
    out_csv: Path,
    run_id: str,
    start_date: str,
    end_date: str,
    created_utc: str,
    table_name: str,
    counts: Dict[str, int],
) -> None:
    """Append many rows per run, e.g. error reasons or impact types."""
    if not counts:
        return
    rows = []
    for k, v in counts.items():
        rows.append(
            {
                "run_id": run_id,
                "start_date": start_date,
                "end_date": end_date,
                "created_utc": created_utc,
                "table": table_name,
                "key": k,
                "count": int(v),
            }
        )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        df.to_csv(out_csv, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(out_csv, index=False, encoding="utf-8")


def compute_and_write_run_stats(
    *,
    run_dir: Path,
    run_id: str,
    start_date: str,
    end_date: str,
    # numbered CSVs from your pipeline:
    step1_csv: Path,  # after download/combined
    step3_csv: Path, # input to first_classifier
    step4_csv: Path,  # after first_classifier outputs flags
    step5_csv: Path,  # remaining after first classifier (optional but useful)
    step6_csv: Path,  # final output of second_classifier
) -> None:
    """
    Writes:
      - run_stats.json (per run)
      - run_stats_runs.csv (append)
      - run_stats_errors.csv (append, long)
      - run_stats_impacts.csv (append, long)
    """
    stats = RunStats(run_id=run_id, start_date=start_date, end_date=end_date)

    # ---------- Step 1: download ----------
    df1 = pd.read_csv(step1_csv)
    stats.set("download_rows_total", int(len(df1)))
    if "sourceurl" in df1.columns:
        stats.set("download_unique_sourceurl", int(df1["sourceurl"].nunique()))

    # ---------- Step 3: input to first classifier ----------
    df3 = pd.read_csv(step3_csv)
    stats.set("zambia_sourceurl", int(len(df3)))

    # ---------- Step 4: first classifier ----------
    df4 = pd.read_csv(step4_csv)
    stats.set("after_first_classifier_rows_total", int(len(df4)))

    removed_first = _count_true(df4, "definitely not mining")
    if removed_first is not None:
        stats.set("first_classifier_removed_definitely_not_mining", int(removed_first))
        stats.set("first_classifier_kept", int(len(df4) - removed_first))

    # ---------- Step 5: remaining (post first classifier) ----------
    if step5_csv.exists():
        df5 = pd.read_csv(step5_csv)
        stats.set("post_first_classifier_remaining_rows_total", int(len(df5)))

    # ---------- Step 6: second classifier final ----------
    df6 = pd.read_csv(step6_csv)
    stats.set("final_rows_total", int(len(df6)))
    # --- Zambia filter view for stats (only if column exists) ---
    zambia_col = _col_present(df6, "in_zambia", "is_in_zambia")
    df6_zm = df6  # default: no filtering if column missing

    if zambia_col:
        zambia_mask = _safe_bool_series(df6, zambia_col)
        if zambia_mask is not None:
            df6_zm = df6[zambia_mask].copy()
            stats.set("second_classifier_in_zambia", int(len(df6_zm)))
            # optional: rate
            stats.set("second_classifier_in_zambia_rate", float(len(df6_zm) / max(1, len(df6))))
    # if "sourceurl" in df6.columns:
    #     stats.set("final_unique_sourceurl", int(df6["sourceurl"].nunique()))
    #     stats.set("final_top_domains_json", _domain_counts(df6, "sourceurl", top_n=50))

    # (a) how many were mining related after second classifier?
    # Try common column names; if none, we leave it out.
    mining_col = _col_present(df6_zm, "is_mining_related", "mining_related", "mining_related_bool", "mining_relatedness")
    if mining_col:
        if mining_col == "mining_relatedness":
            # expects yes/no type values
            s = df6_zm[mining_col].fillna("").astype(str).str.strip().str.lower()
            stats.set("second_classifier_mining_related", int((s == "yes").sum()))
        else:
            v = _count_true(df6_zm, mining_col)
            if v is not None:
                stats.set("second_classifier_mining_related", int(v))

    # (b) webscraping errors + reasons
    err_col = _col_present(df6_zm, "fetch_error", "scrape_error", "article_fetch_error")
    if err_col:
        err = df6_zm[err_col].fillna("").astype(str).str.strip()
        stats.set("second_classifier_rows_with_scrape_error", int((err != "").sum()))
        reasons = _error_reason_counts(df6_zm, err_col, top_n=100)
        # store in JSON for quick inspection
        stats.set("second_classifier_scrape_error_reasons_json", reasons)
        # also write long format table
        write_long_table(
            out_csv=run_dir / "run_stats_errors.csv",
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
            created_utc=stats.created_utc,
            table_name="scrape_error_reason",
            counts=reasons,
        )

    # (c) impacts by focus: Environmental / Social / Governance
    # Adjust candidates to your real column names if needed.
    impact_focus_col = _col_present(df6_zm, "impact_focus", "impact_level1", "impact_EGS", "impact_category")
    if impact_focus_col:
        # Count ONLY rows that have any impacts in this column
        raw = df6_zm[impact_focus_col].fillna("").astype(str).str.strip()
        has_any_impacts = raw.replace({"": "UNKNOWN"}).ne("UNKNOWN")
        stats.set("rows_with_any_impacts", int(has_any_impacts.sum()))

        # Count individual categories, splitting on "||"
        # Example: "Social || Governance || Governance" -> Social:1, Governance:2
        cat_counts: Dict[str, int] = {}
        for s in raw[has_any_impacts]:
            parts = [p.strip() for p in s.split("||")]
            for p in parts:
                if not p:
                    continue
                cat_counts[p] = cat_counts.get(p, 0) + 1

        # Save to JSON (handy for quick inspection)
        stats.set(f"impact_category_counts_from_{impact_focus_col}_json", cat_counts)

        # Write long format table (one row per category) — only if any impacts exist
        write_long_table(
            out_csv=run_dir / "run_stats_impacts.csv",
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
            created_utc=stats.created_utc,
            table_name=f"impact_category_from_{impact_focus_col}",
            counts=cat_counts,
        )

    # Extra useful stats: empty title/desc rates
    if "title" in df6_zm.columns:
        has_title = df6_zm["title"].fillna("").astype(str).str.strip() != ""
        stats.set("final_has_title_rate", float(has_title.mean()))
    if "description" in df6_zm.columns:
        has_desc = df6_zm["description"].fillna("").astype(str).str.strip() != ""
        stats.set("final_has_description_rate", float(has_desc.mean()))

    # ---------- Write outputs ----------
    stats.dump_json(run_dir / "run_stats.json")
    stats.append_runs_csv(run_dir / "run_stats_runs.csv")

import html
import re
from ftfy import fix_text

def clean_text(s):
    if s is None:
        return ""
    s = html.unescape(str(s))
    s = fix_text(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


INTERIM_GDELT_DIR = PROJECT_ROOT / "data" / "interim" / "gdelt_event_context_daily"

def _parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def _prompt_date(label: str) -> date:
    while True:
        raw = input(f"Enter {label} (YYYY-MM-DD): ").strip()
        try:
            return _parse_date(raw)
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD (e.g., 2026-03-01).")


def _date_range(start: date, end: date) -> List[date]:
    if end < start:
        raise ValueError("end_date must be >= start_date")
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def _daily_events_path(d: date) -> Path:
    # matches download.py daily_output_path
    return (
        INTERIM_GDELT_DIR
        / d.strftime("%Y")
        / d.strftime("%m")
        / d.strftime("%d")
        / f"{d.strftime('%Y%m%d')}_events_full.csv"
    )


def download_range(start: date, end: date) -> List[Path]:
    paths: List[Path] = []
    for d in _date_range(start, end):
        day_str = d.strftime("%Y%m%d")
        print(f"\n[download] {day_str}")
        download_stage.main(day_str)
        p = _daily_events_path(d)
        if not p.exists():
            raise FileNotFoundError(f"Expected daily export not found after download: {p}")
        paths.append(p)
    return paths


def combine_daily_exports(daily_paths: List[Path], out_path: Path) -> Path:
    print(f"\n[combine] Combining {len(daily_paths)} daily files...")

    if not daily_paths:
        raise ValueError("No daily files provided to combine_daily_exports.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_file = True
    total_rows = 0
    n_cols = None

    for i, p in enumerate(daily_paths, start=1):
        print(f"[combine] Reading {i}/{len(daily_paths)}: {p.name}")

        df = pd.read_csv(p, low_memory=False)
        total_rows += len(df)

        if n_cols is None:
            n_cols = df.shape[1]

        df.to_csv(
            out_path,
            mode="w" if first_file else "a",
            header=first_file,
            index=False,
            encoding="utf-8",
        )

        first_file = False

        del df
        gc.collect()

    print(f"Saved: {out_path}  (rows={total_rows:,}, cols={n_cols})")
    return out_path


@dataclass
class PipelinePaths:
    run_dir: Path
    events_full: Path
    zambia_collapsed: Path
    enriched: Path
    mining_filtered: Path
    remaining: Path
    final: Path


def make_paths(run_dir: Path) -> PipelinePaths:
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelinePaths(
        run_dir=run_dir,
        events_full=run_dir / "01_events_full_combined.csv",
        zambia_collapsed=run_dir / "02_zambia_events_collapsed.csv",
        enriched=run_dir / "03_zambia_events_enriched.csv",
        mining_filtered=run_dir / "04_mining_filtered.csv",
        remaining=run_dir / "05_remaining.csv",
        final=run_dir / "06_second_classifier_final.csv",
    )


def _get_dates_interactively_if_missing(
    start_arg: Optional[str],
    end_arg: Optional[str],
) -> tuple[date, date]:
    # If both provided, just parse + validate
    if start_arg and end_arg:
        start = _parse_date(start_arg)
        end = _parse_date(end_arg)
        if end < start:
            raise ValueError("end_date must be >= start_date")
        return start, end

    # Otherwise prompt step-by-step
    start = _parse_date(start_arg) if start_arg else _prompt_date("START date")
    while True:
        end = _parse_date(end_arg) if end_arg else _prompt_date("END date")
        if end < start:
            print("End date must be on or after start date. Please enter again.")
            end_arg = None
            continue
        return start, end


def main() -> None:
    ap = argparse.ArgumentParser()
    # now optional; if omitted we prompt
    ap.add_argument("--start-date", required=False, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=False, help="YYYY-MM-DD")

    ap.add_argument("--run-dir", default=None, help="Directory to write numbered intermediate outputs")
    ap.add_argument("--model", default=None, help="Override OpenAI model for classifiers")
    ap.add_argument("--max-rows", type=int, default=None, help="Debug: limit rows passed to classifiers")

    args = ap.parse_args()

    start, end = _get_dates_interactively_if_missing(args.start_date, args.end_date)

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else PROJECT_ROOT / "data" / "processed" / "pipeline_runs" / f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    )
    # start_date = args.start_date
    # end_date = args.end_date
    # run_id = f"{start_date}_{end_date}"
    start_date = start.isoformat()
    end_date = end.isoformat()
    run_id = f"{start_date}_{end_date}"
    paths = make_paths(run_dir)

    print(f"\nRunning pipeline for: {start.isoformat()} → {end.isoformat()}")
    print(f"Run directory: {paths.run_dir}")

    # 1) download
    daily_paths = download_range(start, end)

    # 1b) combine
    combine_daily_exports(daily_paths, paths.events_full)

    # 2) analyse_raw
    print("\n[analyse_raw] filtering Zambia + collapsing by sourceurl")
    run_analyse_raw(paths.events_full, paths.zambia_collapsed)
    print(f"Saved: {paths.zambia_collapsed}")

    # 3) fetch_metadata
    print("\n[fetch_metadata] enriching with title/description")
    enrich_file(paths.zambia_collapsed, paths.enriched)
    print(f"Saved: {paths.enriched}")

    # 4) first_classifier
    model = args.model  # None => each module default, but we keep your explicit default below

    print("\n[first_classifier] definitely-not-mining filter")
    run_filter(
        in_path=paths.enriched,
        out_path=paths.mining_filtered,
        # model=model or "gpt-5-nano-2025-08-07",
        model=model or "gpt-5-mini",
        max_rows=args.max_rows,
    )
    print(f"Saved: {paths.mining_filtered}")

    print("\n[first_classifier] keeping remaining rows")
    keep_remaining(paths.mining_filtered, paths.remaining)
    print(f"Saved: {paths.remaining}")

    # 5) second_classifier
    print("\n[second_classifier] scrape + mining impacts")
    run_stage2(
        in_path=paths.remaining,
        out_path=paths.final,
        # model=model or "gpt-5-nano-2025-08-07",
        model = model or "gpt-5-mini",
        max_rows=args.max_rows,
        scrape_cache_path=paths.final.with_name(paths.final.stem + "_scrape_cache.csv"),
    )
    print(f"Saved: {paths.final}")

    print("\nPipeline complete.")
    print(f"Final output: {paths.final}")

    # Define numbered output paths (must match your pipeline filenames exactly)
    step1_csv = run_dir / "01_events_full_combined.csv"
    step3_csv = run_dir / "03_zambia_events_enriched.csv"
    step4_csv = run_dir / "04_mining_filtered.csv"
    step5_csv = run_dir / "05_remaining.csv"
    step6_csv = run_dir / "06_second_classifier_final.csv"

    compute_and_write_run_stats(
        run_dir=run_dir,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        step1_csv=step1_csv,
        step3_csv=step3_csv,
        step4_csv=step4_csv,
        step5_csv=step5_csv,
        step6_csv=step6_csv,
    )

    print(f"Stats written: {run_dir / 'run_stats.json'}")
    print(f"Run log appended: {run_dir / 'run_stats_runs.csv'}")

    final_csv = run_dir / "final.csv"

    build_final_dataset(
        step6_csv=step6_csv,
        out_csv=final_csv,
    )
    


if __name__ == "__main__":
    main()


## for one week (8 days actually) (20260215 - 20260222):
#### cost = , time = ,
#### is the donwloading slowing down because of memory usage?
#### i want the date to come through to the end. 
### downloading took about 5 minutes and then the first classifier took the rest of the time. 

