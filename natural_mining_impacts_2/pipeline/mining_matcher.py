from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
SCRAPE_DIR = BASE_DIR / "data" / "processed" / "webscraped_daily"
MATCH_DIR = BASE_DIR / "data" / "processed" / "mining_matched_daily"
STATE_DIR = BASE_DIR / "data" / "interim" / "_state"

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
COMPILED_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in MINING_PATTERNS]


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    if pd.isna(v):
        return ""
    return str(v).strip()


def _build_haystack(title: str, description: str, text: str) -> str:
    return " ".join([
        _norm_str(title),
        _norm_str(description),
        _norm_str(text),
    ]).strip()


def matched_mining_terms(title: str, description: str, text: str) -> list[str]:
    """
    Return the actual matched words/phrases found in the text,
    not just the regex patterns.
    """
    haystack = _build_haystack(title, description, text)
    if not haystack:
        return []

    matches = []

    for pat in COMPILED_PATTERNS:
        found = pat.findall(haystack)

        if not found:
            continue

        for m in found:
            if isinstance(m, tuple):
                m = " ".join(str(x) for x in m if x)

            m = str(m).strip()
            if m:
                matches.append(m)

    # deduplicate while preserving order, and lowercase for consistency
    deduped = []
    seen = set()
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    return deduped


def run_mining_matcher(
    in_path: Path,
    out_path: Path,
    max_rows: Optional[int] = None,
    include_match_details: bool = True,
) -> Path:
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8", engine="python")
    if max_rows:
        df = df.head(max_rows).copy()

    if "url_normalized" not in df.columns:
        raise ValueError("Expected 'url_normalized' column.")

    keep_rows = []

    for _, r in df.iterrows():
        row = r.to_dict()

        title = _norm_str(row.get("title"))
        description = _norm_str(row.get("meta_description", row.get("description", "")))
        text = _norm_str(row.get("text"))

        haystack = _build_haystack(title, description, text)
        nonempty_text = bool(haystack)
        keyword_hit = bool(COMPILED_MINING_REGEX.search(haystack)) if nonempty_text else False

        scrape_ok_raw = row.get("scrape_ok", False)
        scrape_ok = (
            str(scrape_ok_raw).lower() in {"true", "1", "yes"}
            if not isinstance(scrape_ok_raw, bool)
            else scrape_ok_raw
        )
        scrape_failed = not scrape_ok

        row["mining_keyword_score"] = 1 if keyword_hit else 0
        row["mining_keyword_match"] = bool(keyword_hit or scrape_failed)
        row["mining_keyword_force_keep"] = bool(scrape_failed)

        if include_match_details and keyword_hit:
            matches = matched_mining_terms(title, description, text)
            row["mining_keyword_match_detail"] = "; ".join(matches)
            row["mining_keyword_matches"] = "|".join(matches)
        else:
            row["mining_keyword_match_detail"] = ""
            row["mining_keyword_matches"] = ""

        if scrape_failed:
            row["mining_keyword_decision_reason"] = "scrape_failed_force_keep"
        elif keyword_hit:
            row["mining_keyword_decision_reason"] = "keyword_match"
        else:
            row["mining_keyword_decision_reason"] = "filtered_out"

        if row["mining_keyword_match"]:
            keep_rows.append(row)

    out_df = pd.DataFrame(keep_rows)

    if "mining_keyword_score" in out_df.columns and len(out_df) > 0:
        out_df["mining_keyword_score"] = out_df["mining_keyword_score"].astype("int8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    total_in = len(df)
    total_kept = len(out_df)
    kept_keyword = int((out_df["mining_keyword_decision_reason"] == "keyword_match").sum()) if total_kept else 0
    kept_scrape_failed = int((out_df["mining_keyword_decision_reason"] == "scrape_failed_force_keep").sum()) if total_kept else 0
    filtered_out = total_in - total_kept

    print("[mining_matcher] keyword mining filter using shared webscraper output")
    print(f"Saved: {out_path}")
    print(f"Input rows: {total_in:,}")
    print(f"Kept by keyword: {kept_keyword:,}")
    print(f"Kept due to scrape failure: {kept_scrape_failed:,}")
    print(f"Filtered out: {filtered_out:,}")
    print(f"Rows kept after mining matcher: {total_kept:,}")

    return out_path


def main(target_date: str, force: bool = False, max_rows: Optional[int] = None) -> None:
    year, month, day = target_date[:4], target_date[4:6], target_date[6:8]

    in_path = SCRAPE_DIR / year / month / day / f"{target_date}_webscraped.csv"
    out_dir = MATCH_DIR / year / month / day
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{target_date}_mining_matched.csv"
    state_out = STATE_DIR / f"mining_matched_{target_date}.csv"

    if out_path.exists() and not force:
        print(f"Skipping mining matcher for {target_date}: already done ({out_path})")
        return

    run_mining_matcher(
        in_path=in_path,
        out_path=out_path,
        max_rows=max_rows,
        include_match_details=True,
    )

    df = pd.read_csv(out_path, encoding="utf-8", engine="python")
    df.to_csv(state_out, index=False)

    print(f"State CSV: {state_out}\n")


if __name__ == "__main__":
    target_date = input("Enter date for mining matcher (YYYYMMDD): ").strip()
    main(target_date, force=False)