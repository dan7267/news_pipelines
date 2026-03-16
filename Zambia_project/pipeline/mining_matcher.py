from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

MIN_MATCH_SCORE = 1

MINING_PATTERNS = [
    r"\bmine\b",
    r"\bmines\b",
    r"\bmining\b",
    r"\bminer\b",
    r"\bmineral\b",
    r"\bminerals\b",
    r"\braw\b",
    r"\bmaterial\b",
    r"\bmaterials\b",
    r"\bore\b",
    r"\bsmelter\b",
    r"\bsmelting\b",
    r"\btailings?\b",
    r"\bopen[- ]?pit\b",
    r"\bshaft\b",
    r"\bdrill(?:ing)?\b",
    r"\broyalt(?:y|ies)\b",
    r"\blicen[cs]e\b",
    r"\bpermit\b",
    r"\bcopper\b",
    r"\bcobalt\b",
    r"\bgold\b",
    r"\bnickel\b",
    r"\blithium\b",
    r"\bmanganese\b",
    r"\bcoal\b",
    r"\bgraphite\b",
    r"\buranium\b",
    r"\brare earth(?:s)?\b",
    r"\bconcentrator\b",
    r"\brefiner(?:y)?\b",
    r"\bKonkola\b",
    r"\bKCM\b",
    r"\bMopani\b",
    r"\bKansanshi\b",
    r"\bLumwana\b",
    r"\bFirst Quantum\b",
    r"\bBarrick\b",
    r"\bVedanta\b",
    r"\bZCCM[- ]?IH\b",
    r"\bquarry\b",
    r"\bextract\b",
    r"\bextraction\b",
    r"\brare\b",
    r"\brare-earth\b",
    r"\belements\b",
    r"\bcopperbelt\b",
    r"\bmetal\b",
    r"\bmetals\b",
    r"\bmineworker\b",
    r"\bmineworkers\b",
]

COMPILED_MINING_REGEX = re.compile("|".join(MINING_PATTERNS), flags=re.IGNORECASE)


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _build_text_series(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["title", "description", "text"]:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))

    if not parts:
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    text = parts[0]
    for part in parts[1:]:
        text = text + " " + part
    return text.str.strip()


def run_mining_matcher(
    in_path: Path,
    out_path: Path,
    scrape_cache_path: Path,
    max_rows: Optional[int] = None,
) -> Path:
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    if max_rows:
        df = df.head(max_rows).copy()

    if "sourceurl" not in df.columns:
        raise ValueError("Expected sourceurl column.")

    # Keep signature-compatible with existing pipeline, but the matcher no longer
    # scrapes or uses the scrape cache because that was a major bottleneck.
    _ = scrape_cache_path

    text = _build_text_series(df)
    nonempty_mask = text.ne("")

    print(f"[diagnostic] total rows: {len(df):,}")
    print(f"[diagnostic] non-empty combined text rows: {int(nonempty_mask.sum()):,}")

    for col in ["title", "description", "text"]:
        if col in df.columns:
            nonempty = (df[col].fillna("").astype(str).str.strip() != "").sum()
            print(f"[diagnostic] {col}: {nonempty:,} non-empty")
        else:
            print(f"[diagnostic] {col}: MISSING")

    df["mining_keyword_score"] = pd.Series(0, index=df.index, dtype="int8")
    if nonempty_mask.any():
        df.loc[nonempty_mask, "mining_keyword_score"] = (
            text.loc[nonempty_mask]
            .str.contains(COMPILED_MINING_REGEX, na=False)
            .astype("int8")
        )
    print(f"[diagnostic] keyword matches: {int((df['mining_keyword_score'] >= MIN_MATCH_SCORE).sum()):,}")    

    df["mining_keyword_match"] = (
        (df["mining_keyword_score"] >= MIN_MATCH_SCORE) | (~nonempty_mask)
    )
    df["mining_keyword_force_keep"] = ~nonempty_mask

    df["mining_keyword_decision_reason"] = "filtered_out"
    df.loc[df["mining_keyword_score"] >= MIN_MATCH_SCORE, "mining_keyword_decision_reason"] = "keyword_match"
    df.loc[~nonempty_mask, "mining_keyword_decision_reason"] = "empty_text_force_keep"

    # Preserve downstream compatibility where these columns may be expected,
    # but keep them blank so we avoid expensive per-row match-detail extraction.
    for col in [
        "scrape_status",
        "scraped_title",
        "scraped_published_date",
        "mining_keyword_match_detail",
        "mining_keyword_matches",
    ]:
        if col not in df.columns:
            df[col] = ""

    out_df = df.loc[df["mining_keyword_match"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Input rows: {len(df):,}")
    print(f"Rows kept after mining matcher: {len(out_df):,}")

    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scrape-cache", required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    run_mining_matcher(
        in_path=Path(args.input),
        out_path=Path(args.output),
        scrape_cache_path=Path(args.scrape_cache),
        max_rows=args.max_rows,
    )
