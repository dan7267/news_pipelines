from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

# reuse stage2 scrape helpers
from second_classifier import (
    load_scrape_cache,
    save_scrape_cache,
    scrape_one,
    normalize_url_basic,
)

MAX_WORKERS = 10
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
    # r"\bexploration\b",
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
    # r"\bcritical\b",
    r"\bextract\b",
    r"\bextraction\b",
    r"\brare\b",
    # r"\bearth\b",
    r"\brare-earth\b",
    # r"\belement\b",
    r"\belements\b",
    r"\bcopperbelt\b",
    # r"\bkitwe\b",
    # r"\bluanshya\b",
    # r"\bmufulira\b",
    # r"\bchingola\b",
    r"\bmetal\b",
    r"\bmetals\b",
    # r"\blead\b",
    r"\bmineworker\b",
    r"\bmineworkers\b",
]

# Fast boolean matcher for the main decision
COMPILED_MINING_REGEX = re.compile("|".join(MINING_PATTERNS), flags=re.IGNORECASE)

# Optional: only used if you later decide you want match details again
COMPILED_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in MINING_PATTERNS]


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _build_haystack(title: str, description: str, text: str) -> str:
    return " ".join([
        _norm_str(title),
        _norm_str(description),
        _norm_str(text),
    ]).strip()


def matched_mining_patterns(title: str, description: str, text: str) -> list[str]:
    """
    Kept for compatibility / optional debugging.
    Not used in the fast path unless explicitly enabled.
    """
    haystack = _build_haystack(title, description, text)
    if not haystack:
        return []

    matches = []
    for pat in COMPILED_PATTERNS:
        if pat.search(haystack):
            matches.append(pat.pattern)
    return matches


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

    scrape_cache = load_scrape_cache(scrape_cache_path)
    rows = df.to_dict(orient="records")

    def scrape_cached(url: str) -> Dict[str, Any]:
        key = normalize_url_basic(url)

        if key in scrape_cache:
            cached = scrape_cache[key]

            cached_ok = str(cached.get("scrape_ok", "")).lower() in {"true", "1", "yes"}
            cached_used_wayback = str(cached.get("used_wayback", "")).lower() in {"true", "1", "yes"}

            # Long-term rule:
            # - reuse successful scrapes
            # - reuse previous Wayback rescues
            # - but DO NOT reuse old failed scrapes, because they may now be recoverable via Wayback
            if cached_ok:
                return {
                    "url_normalized": key,
                    "final_url": cached.get("final_url", url),
                    "scrape_ok": cached_ok,
                    "scrape_status": cached.get("scrape_status", ""),
                    "scrape_error": cached.get("scrape_error", ""),
                    "scraped_title": cached.get("scraped_title", ""),
                    "scraped_published_date": cached.get("scraped_published_date", "unknown"),
                    "text": cached.get("text", ""),
                    "http_status": cached.get("http_status", ""),
                    "used_wayback": cached_used_wayback,
                    "wayback_url": cached.get("wayback_url", ""),
                    "wayback_timestamp": cached.get("wayback_timestamp", ""),
                }

        out = scrape_one(url)

        scrape_cache[key] = {
            "url_normalized": out["url_normalized"],
            "final_url": out.get("final_url", url),
            "scrape_ok": str(out.get("scrape_ok", False)),
            "scrape_status": out.get("scrape_status", ""),
            "scrape_error": out.get("scrape_error", ""),
            "scraped_title": out.get("scraped_title", ""),
            "scraped_published_date": out.get("scraped_published_date", "unknown"),
            "text": out.get("text", ""),
            "http_status": str(out.get("http_status", "")),
            "used_wayback": str(out.get("used_wayback", False)),
            "wayback_url": out.get("wayback_url", ""),
            "wayback_timestamp": out.get("wayback_timestamp", ""),
        }
        return out
    urls = [_norm_str(r.get("sourceurl")) for r in rows]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        scraped = list(
            tqdm(
                ex.map(scrape_cached, urls),
                total=len(urls),
                desc="Scraping for mining matcher",
            )
        )

    save_scrape_cache(scrape_cache_path, scrape_cache)

    keep_rows = []

    for r, sc in zip(rows, scraped):
        title = _norm_str(r.get("title"))
        description = _norm_str(r.get("description"))
        text = _norm_str(sc.get("text"))

        haystack = _build_haystack(title, description, text)
        nonempty_text = bool(haystack)
        keyword_hit = bool(COMPILED_MINING_REGEX.search(haystack)) if nonempty_text else False
        scrape_failed = not bool(sc.get("scrape_ok", False))

        r["scrape_status"] = sc.get("scrape_status", "")
        r["scrape_error"] = sc.get("scrape_error", "")
        r["scraped_title"] = sc.get("scraped_title", "")
        r["scraped_published_date"] = sc.get("scraped_published_date", "unknown")
        r["final_url"] = sc.get("final_url", _norm_str(r.get("sourceurl")))
        r["http_status"] = sc.get("http_status", "")
        r["used_wayback"] = bool(sc.get("used_wayback", False))
        r["wayback_url"] = sc.get("wayback_url", "")
        r["wayback_timestamp"] = sc.get("wayback_timestamp", "")
        r["text"] = text

        # Keep compact and simple
        r["mining_keyword_score"] = 1 if keyword_hit else 0
        r["mining_keyword_match"] = keyword_hit or scrape_failed
        r["mining_keyword_force_keep"] = scrape_failed

        # Preserve downstream compatibility, but avoid expensive detail extraction
        r["mining_keyword_match_detail"] = ""
        r["mining_keyword_matches"] = ""

        if scrape_failed:
            r["mining_keyword_decision_reason"] = "scrape_failed_force_keep"
        elif keyword_hit:
            r["mining_keyword_decision_reason"] = "keyword_match"
        else:
            r["mining_keyword_decision_reason"] = "filtered_out"

        if r["mining_keyword_match"]:
            keep_rows.append(r)

    out_df = pd.DataFrame(keep_rows)

    # Keep dtype compact where possible
    if "mining_keyword_score" in out_df.columns:
        out_df["mining_keyword_score"] = out_df["mining_keyword_score"].astype("int8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[mining_matcher] scrape + keyword mining filter")
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