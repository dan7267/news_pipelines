"""analyse_raw.py

Stage 1 post-download processing.

- Loads one or more GDELT daily exports (combined beforehand)
- Filters to Zambia-related rows (geo + actor + text heuristics)
- Collapses multiple GDELT event rows that share the same sourceurl
  into one row per URL, storing the per-event variants as JSON.

This is a refactor of analyse_raw.ipynb, keeping only logic needed
for downstream stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


ZAMBIA_GEO_FIPS = "ZA"   # GDELT geo country code for Zambia
ZAMBIA_ACTOR_CAMEO = "ZMB"  # CAMEO country code for Zambia (actors)

TEXT_TERMS = [
    "zambia",
    "zambian",
    "lusaka",
    "copperbelt",
    "ndola",
    "kitwe",
    "kabwe",
    "livingstone",
    "chingola",
    "solwezi",
    "chipata",
    "kasama",
    "mongu",
]

# Columns that are not useful once we collapse per-url variants
DROP_COLS = [
    "ingest_time",
    "globaleventid",
    "fractiondate",
    "isrootevent",
    "numentions",
    "numsources",
    "numarticles",
    "dateadded",
]


def contains_any(series: pd.Series, terms: List[str]) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=s.index)
    for t in terms:
        mask = mask | s.str.contains(t, regex=False)
    return mask


def filter_zambia(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows likely to be about Zambia (conservative OR mask)."""

    geo_country_cols = ["actiongeo_countrycode", "actor1geo_countrycode", "actor2geo_countrycode"]
    geo_adm1_cols = ["actiongeo_adm1code", "actor1geo_adm1code", "actor2geo_adm1code"]
    geo_full_cols = ["actiongeo_fullname", "actor1geo_fullname", "actor2geo_fullname"]
    actor_country_cols = ["actor1countrycode", "actor2countrycode"]
    actor_name_cols = ["actor1name", "actor2name"]

    geo_country_hit = pd.Series(False, index=df.index)
    for c in geo_country_cols:
        if c in df.columns:
            geo_country_hit = geo_country_hit | (df[c] == ZAMBIA_GEO_FIPS)

    geo_adm1_hit = pd.Series(False, index=df.index)
    for c in geo_adm1_cols:
        if c in df.columns:
            geo_adm1_hit = geo_adm1_hit | df[c].fillna("").astype(str).str.startswith(ZAMBIA_GEO_FIPS)

    actor_country_hit = pd.Series(False, index=df.index)
    for c in actor_country_cols:
        if c in df.columns:
            actor_country_hit = actor_country_hit | (df[c] == ZAMBIA_ACTOR_CAMEO)

    geo_text_hit = pd.Series(False, index=df.index)
    for c in geo_full_cols:
        if c in df.columns:
            geo_text_hit = geo_text_hit | contains_any(df[c], TEXT_TERMS)

    actor_text_hit = pd.Series(False, index=df.index)
    for c in actor_name_cols:
        if c in df.columns:
            actor_text_hit = actor_text_hit | contains_any(df[c], ["zambia", "zambian"])

    zambia_mask = geo_country_hit | geo_adm1_hit | actor_country_hit | geo_text_hit | actor_text_hit
    return df[zambia_mask].copy()


def _json_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetimes + NaNs to JSON-safe python types."""
    out = df.copy()

    dt_cols = out.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for c in dt_cols:
        out[c] = out[c].dt.strftime("%Y-%m-%d")

    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)

    out = out.replace({np.nan: "", pd.NaT: ""})
    return out


def collapse_by_sourceurl_store_variants(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sourceurl; store all per-event variants as JSON; keep one sqldate."""

    keep_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").copy()
    if "sourceurl" not in keep_df.columns:
        raise ValueError("Expected a 'sourceurl' column.")
    if "sqldate" not in keep_df.columns:
        raise ValueError("Expected a 'sqldate' column.")

    # Make sqldate comparable and stable (string is fine; int also fine)
    keep_df["sqldate"] = keep_df["sqldate"].astype(str)

    variant_cols = [c for c in keep_df.columns if c != "sourceurl"]
    keep_df = _json_safe_df(keep_df)

    def build_variants(g: pd.DataFrame) -> str:
        unique_rows = g[variant_cols].drop_duplicates()
        variants = unique_rows.to_dict(orient="records")
        return json.dumps(variants, ensure_ascii=False, sort_keys=True)

    collapsed = (
        keep_df.groupby("sourceurl")
        .apply(
            lambda g: pd.Series(
                {
                    # choose ONE sqldate for this sourceurl (earliest)
                    "sqldate": g["sqldate"].min(),

                    "n_variants": g[variant_cols].drop_duplicates().shape[0],
                    "variants_json": build_variants(g),
                }
            )
        )
        .reset_index()
    )

    return collapsed


def run_analyse_raw(
    in_path: Path,
    out_path: Path,
) -> Path:
    """Load combined events CSV -> filter Zambia -> collapse per-url."""

    df = pd.read_csv(in_path)

    df_zambia = filter_zambia(df)
    df_collapsed = collapse_by_sourceurl_store_variants(df_zambia)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_collapsed.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    p = run_analyse_raw(Path(args.in_path), Path(args.out_path))
    print(f"Saved: {p}")
