"""
summarise_event_codes.py

Load:
  /data/processed/pipeline_runs/20260218_20260304/final.csv

Take all sourceurl rows from final.csv, look them up in:
  /data/processed/pipeline_runs/20260218_20260304/01_events_full_combined

Then compute summary statistics for:
  - eventcode
  - eventbasecode
  - eventrootcode
  - quadclass
  - goldsteinscale

Outputs:
  - matched_events.csv
  - overall_code_counts.csv
  - overall_goldsteinscale_summary.csv
  - per_sourceurl_code_counts.csv
  - per_sourceurl_goldsteinscale_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_FINAL = Path("data/processed/pipeline_runs/20260218_20260304/final.csv")
DEFAULT_EVENTS = Path("data/processed/pipeline_runs/20260218_20260304/01_events_full_combined")
DEFAULT_OUTDIR = Path("data/processed/pipeline_runs/20260218_20260304/eventcode_summary")

CODE_COLUMNS = ["eventcode", "eventbasecode", "eventrootcode", "quadclass"]
NUMERIC_COLUMNS = ["goldsteinscale"]

def load_table(path_like: Path) -> pd.DataFrame:
    """
    Load a table from a flexible path.
    Supports:
      - exact file path
      - path stem without extension, trying common tabular extensions
    """
    candidates = []

    if path_like.exists():
        candidates.append(path_like)

    if not candidates:
        for ext in [".csv", ".tsv", ".parquet", ".pkl", ".pickle"]:
            candidate = path_like.with_suffix(ext)
            if candidate.exists():
                candidates.append(candidate)

    if not candidates and path_like.is_dir():
        raise ValueError(
            f"{path_like} is a directory. Please point to a file "
            f"(e.g. CSV/TSV/Parquet), not a folder."
        )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find input table for: {path_like}\n"
            f"Tried exact path plus extensions: .csv, .tsv, .parquet, .pkl, .pickle"
        )

    path = candidates[0]
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(f"Unsupported file type: {path}")


def normalise_url_series(s: pd.Series) -> pd.Series:
    """
    Light URL normalisation for matching:
      - cast to string
      - strip whitespace
      - lower-case
      - remove trailing slash
    """
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.rstrip("/")
    )


def require_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {df_name}: {missing}")


def overall_code_counts(df: pd.DataFrame, code_columns: list[str]) -> pd.DataFrame:
    """
    Produce one long table of counts for each categorical code column.
    """
    out_frames = []
    for col in code_columns:
        vc = (
            df[col]
            .fillna("<<MISSING>>")
            .value_counts(dropna=False)
            .rename_axis("code_value")
            .reset_index(name="count")
        )
        vc.insert(0, "code_column", col)
        vc["share"] = vc["count"] / len(df) if len(df) else 0.0
        out_frames.append(vc)

    if not out_frames:
        return pd.DataFrame(columns=["code_column", "code_value", "count", "share"])

    return pd.concat(out_frames, ignore_index=True)


def per_sourceurl_code_counts(df: pd.DataFrame, code_columns: list[str]) -> pd.DataFrame:
    """
    Produce per-sourceurl counts for each categorical code column.
    """
    out_frames = []
    for col in code_columns:
        grp = (
            df.assign(_code_value=df[col].fillna("<<MISSING>>"))
              .groupby(["sourceurl", "_code_value"], dropna=False)
              .size()
              .reset_index(name="count")
              .rename(columns={"_code_value": "code_value"})
        )

        totals = (
            df.groupby("sourceurl", dropna=False)
              .size()
              .reset_index(name="sourceurl_total_rows")
        )

        grp = grp.merge(totals, on="sourceurl", how="left")
        grp["share_within_sourceurl"] = grp["count"] / grp["sourceurl_total_rows"]
        grp.insert(1, "code_column", col)
        out_frames.append(grp)

    if not out_frames:
        return pd.DataFrame(
            columns=[
                "sourceurl", "code_column", "code_value",
                "count", "sourceurl_total_rows", "share_within_sourceurl"
            ]
        )

    return pd.concat(out_frames, ignore_index=True)


def overall_numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """
    Summary stats for numeric columns.
    """
    rows = []
    for col in numeric_columns:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "column": col,
            "count_non_null": int(s.notna().sum()),
            "count_null": int(s.isna().sum()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "max": s.max(),
        })
    return pd.DataFrame(rows)


def per_sourceurl_numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """
    Per-sourceurl summary stats for numeric columns.
    """
    out_frames = []

    for col in numeric_columns:
        temp = df[["sourceurl", col]].copy()
        temp[col] = pd.to_numeric(temp[col], errors="coerce")

        grp = temp.groupby("sourceurl")[col].agg(
            count_non_null="count",
            mean="mean",
            std="std",
            min="min",
            median="median",
            max="max",
        ).reset_index()

        q25 = temp.groupby("sourceurl")[col].quantile(0.25).reset_index(name="p25")
        q75 = temp.groupby("sourceurl")[col].quantile(0.75).reset_index(name="p75")
        nulls = temp.groupby("sourceurl")[col].apply(lambda s: s.isna().sum()).reset_index(name="count_null")

        grp = grp.merge(q25, on="sourceurl", how="left")
        grp = grp.merge(q75, on="sourceurl", how="left")
        grp = grp.merge(nulls, on="sourceurl", how="left")
        grp.insert(1, "column", col)

        out_frames.append(grp)

    if not out_frames:
        return pd.DataFrame(
            columns=[
                "sourceurl", "column", "count_non_null", "count_null",
                "mean", "std", "min", "p25", "median", "p75", "max"
            ]
        )

    return pd.concat(out_frames, ignore_index=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    final_path: Path,
    events_path: Path,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading final.csv from: {final_path}")
    final_df = load_table(final_path)

    print(f"Loading events table from: {events_path}")
    events_df = load_table(events_path)

    require_columns(final_df, ["sourceurl"], "final_df")
    require_columns(events_df, ["sourceurl"] + CODE_COLUMNS + NUMERIC_COLUMNS, "events_df")

    # Normalise URLs for matching
    final_urls = (
        final_df["sourceurl"]
        .dropna()
        .pipe(normalise_url_series)
    )
    final_url_set = set(final_urls.tolist())

    print(f"Unique sourceurls in final.csv: {len(final_url_set):,}")

    events_df = events_df.copy()
    events_df["sourceurl"] = normalise_url_series(events_df["sourceurl"])

    matched_df = events_df[events_df["sourceurl"].isin(final_url_set)].copy()

    print(f"Matched event rows: {len(matched_df):,}")
    print(f"Matched unique sourceurls: {matched_df['sourceurl'].nunique():,}")

    # Save matched rows
    matched_path = outdir / "matched_events.csv"
    matched_df.to_csv(matched_path, index=False)

    # Overall summaries
    overall_codes = overall_code_counts(matched_df, CODE_COLUMNS)
    overall_goldstein = overall_numeric_summary(matched_df, NUMERIC_COLUMNS)

    overall_codes.to_csv(outdir / "overall_code_counts.csv", index=False)
    overall_goldstein.to_csv(outdir / "overall_goldsteinscale_summary.csv", index=False)

    # Per-sourceurl summaries
    per_url_codes = per_sourceurl_code_counts(matched_df, CODE_COLUMNS)
    per_url_goldstein = per_sourceurl_numeric_summary(matched_df, NUMERIC_COLUMNS)

    per_url_codes.to_csv(outdir / "per_sourceurl_code_counts.csv", index=False)
    per_url_goldstein.to_csv(outdir / "per_sourceurl_goldsteinscale_summary.csv", index=False)

    # Optional compact console summary
    print("\nTop overall code counts:")
    for col in CODE_COLUMNS:
        print(f"\n{col}:")
        sub = overall_codes[overall_codes["code_column"] == col].head(10)
        print(sub.to_string(index=False))

    print("\nOverall goldsteinscale summary:")
    print(overall_goldstein.to_string(index=False))

    print(f"\nDone. Outputs written to: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise event code statistics for sourceurls present in final.csv"
    )
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL, help="Path to final.csv")
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS,
        help="Path to 01_events_full_combined (csv/tsv/parquet/pickle or stem without extension)",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    args = parser.parse_args()

    main(
        final_path=args.final,
        events_path=args.events,
        outdir=args.outdir,
    )