"""fetch_metadata.py

Stage 2: enrich each sourceurl with (best-effort) HTML metadata: title + description.

Refactor of fetch_metadata.ipynb.

Inputs: CSV with at least column `sourceurl`.
Outputs: same rows + columns: title, description, http_status, fetch_error.
Also writes/uses a cache CSV to avoid repeated requests.
"""

from __future__ import annotations

import csv
import html
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from ftfy import fix_text
from tqdm import tqdm

import csv
import sys


def set_max_csv_field_size() -> None:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 10


set_max_csv_field_size()

USER_AGENT = "Mozilla/5.0 (compatible; CCML/1.0)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

TIMEOUT_S = 20
MAX_RETRIES = 2
SLEEP_BETWEEN_REQ = (0.05, 0.15)
MAX_WORKERS = 20

MAX_TITLE_CHARS = 300
MAX_DESC_CHARS = 800

CACHE_FIELDS = ["url_normalized", "title", "description", "http_status", "fetch_error"]


# ------------------ STRONGER TEXT CLEANING ------------------ #

EM_DASH_BAD = "\u00e2\u20ac\u201d"
EN_DASH_BAD = "\u00e2\u20ac\u201c"

CONTROL_MAP = {
    "â€\x91": "‘",
    "â€\x92": "’",
    "â€\x93": "–",
    "â€\x94": "—",
    "â€\x85": "…",
    "â€\x9c": "“",
    "â€\x9d": "”",
}

LITERAL_MAP = {
    "â€“": "–",
    "â€”": "—",
    "â€˜": "‘",
    "â€™": "’",
    "â€œ": "“",
    "â€": "”",
    "â€¦": "…",
    "â„¢": "™",
    "Â£": "£",
    "Â€": "€",
    "Â ": " ",
    "\u00A0": " ",
}


def _try_redecode(s: str) -> str:
    """
    Best-effort repair for text that looks like UTF-8 decoded as cp1252/latin-1.
    Only returns the repaired version if it reduces suspicious mojibake markers.
    """
    suspicious = any(c in s for c in ("â", "Â", "Ã"))
    if not suspicious:
        return s

    original_score = sum(s.count(c) for c in ("â", "Â", "Ã"))

    for enc in ("cp1252", "latin-1"):
        try:
            repaired = s.encode(enc, errors="strict").decode("utf-8", errors="strict")
            repaired_score = sum(repaired.count(c) for c in ("â", "Â", "Ã"))
            if repaired_score < original_score:
                return repaired
        except Exception:
            pass

    return s


def clean_meta_str(x: Optional[str]) -> str:
    """
    Comprehensive metadata cleaner:
    - handles empty / NaN values
    - HTML-unescapes
    - runs ftfy
    - applies manual mojibake maps
    - attempts cp1252/latin-1 -> utf-8 re-decode when suspicious
    - normalizes whitespace
    """
    if x is None or pd.isna(x):
        return ""

    s = str(x)
    if not s:
        return ""

    # 1) HTML entity decoding
    s = html.unescape(s)

    # 2) Generic text fixing
    s = fix_text(s)

    # 3) Explicit bad dash forms
    s = s.replace(EM_DASH_BAD, "—").replace(EN_DASH_BAD, "–")

    # 4) Manual replacement maps
    for bad, good in CONTROL_MAP.items():
        s = s.replace(bad, good)
    for bad, good in LITERAL_MAP.items():
        s = s.replace(bad, good)

    # 5) Try more aggressive re-decode if suspicious junk remains
    if any(c in s for c in ("â", "Â", "Ã")):
        s2 = _try_redecode(s)
        if s2 != s:
            s = fix_text(s2)
            s = s.replace(EM_DASH_BAD, "—").replace(EN_DASH_BAD, "–")
            for mapping in (CONTROL_MAP, LITERAL_MAP):
                for bad, good in mapping.items():
                    s = s.replace(bad, good)

    # 6) Normalize whitespace
    s = " ".join(s.split())
    return s


# ------------------ URL NORMALIZATION ------------------ #

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    try:
        p = urlparse(url)
        q = [
            (k, v)
            for k, v in parse_qsl(p.query, keep_blank_values=True)
            if k.lower()
            not in {
                "gclid",
                "fbclid",
                "mc_cid",
                "mc_eid",
                "igshid",
                "spm",
                "ref",
                "ref_src",
            }
            and not any(k.lower().startswith(pref) for pref in ("utm_",))
        ]
        return urlunparse(
            (p.scheme or "http", p.netloc.lower(), p.path, p.params, urlencode(q, doseq=True), "")
        )
    except Exception:
        return url


def truncate(s: Optional[str], n: int) -> str:
    s = clean_meta_str(s)
    return s[:n] if len(s) > n else s


# ------------------ HTML METADATA EXTRACTION ------------------ #

def _meta_content(soup: BeautifulSoup, *, name: str = "", prop: str = "") -> str:
    tag = None
    if name:
        tag = soup.find("meta", attrs={"name": name})
    if tag is None and prop:
        tag = soup.find("meta", attrs={"property": prop})
    if tag is None:
        return ""
    return clean_meta_str(tag.get("content", ""))


def fetch_title_desc(session: requests.Session, url: str) -> Dict[str, str]:
    """Fetch URL and return a dict with title/description + http_status/fetch_error."""
    out = {"title": "", "description": "", "http_status": "", "fetch_error": ""}

    url_norm = normalize_url(url)

    for attempt in range(MAX_RETRIES + 1):
        try:
            time.sleep(random.uniform(*SLEEP_BETWEEN_REQ))
            r = session.get(url_norm, timeout=TIMEOUT_S, allow_redirects=True)
            out["http_status"] = str(getattr(r, "status_code", ""))

            if r.status_code >= 400:
                out["fetch_error"] = f"HTTP {r.status_code}"
                return out

            soup = BeautifulSoup(r.text or "", "html.parser")

            # Prefer the HTML <title> first
            title = soup.title.string if soup.title and soup.title.string else ""
            title = clean_meta_str(title)

            # Fallbacks for title
            if not title:
                title = _meta_content(soup, prop="og:title") or _meta_content(soup, name="twitter:title")

            # Description fallbacks
            desc = (
                _meta_content(soup, name="description")
                or _meta_content(soup, prop="og:description")
                or _meta_content(soup, name="twitter:description")
            )

            out["title"] = truncate(title, MAX_TITLE_CHARS)
            out["description"] = truncate(desc, MAX_DESC_CHARS)
            return out

        except Exception as e:
            out["fetch_error"] = f"{type(e).__name__}: {e}"
            if attempt >= MAX_RETRIES:
                return out

    return out


# ------------------ CACHE HANDLING ------------------ #

def load_cache(cache_path: Path) -> Dict[str, Dict[str, str]]:
    cache: Dict[str, Dict[str, str]] = {}
    if not cache_path.exists():
        return cache

    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            urln = (row.get("url_normalized") or "").strip()
            if urln:
                cache[urln] = {
                    "url_normalized": urln,
                    "title": clean_meta_str(row.get("title", "")),
                    "description": clean_meta_str(row.get("description", "")),
                    "http_status": str(row.get("http_status", "") or ""),
                    "fetch_error": str(row.get("fetch_error", "") or ""),
                }

    return cache


def save_cache(cache_path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        w.writeheader()
        for _, row in cache.items():
            w.writerow(
                {
                    "url_normalized": row.get("url_normalized", ""),
                    "title": clean_meta_str(row.get("title", "")),
                    "description": clean_meta_str(row.get("description", "")),
                    "http_status": row.get("http_status", ""),
                    "fetch_error": row.get("fetch_error", ""),
                }
            )


# ------------------ PER-ROW PROCESSING ------------------ #

def process_row(row: Dict[str, str], cache: Dict[str, Dict[str, str]], session: requests.Session) -> Dict[str, str]:
    url = (row.get("sourceurl") or "").strip()
    urln = normalize_url(url)

    if urln in cache:
        cached = cache[urln]
        row["title"] = clean_meta_str(cached.get("title", ""))
        row["description"] = clean_meta_str(cached.get("description", ""))
        row["http_status"] = cached.get("http_status", "")
        row["fetch_error"] = cached.get("fetch_error", "")
        return row

    meta = fetch_title_desc(session, url)
    row.update(meta)

    # Ensure cleaned even if future logic changes upstream
    row["title"] = clean_meta_str(row.get("title", ""))
    row["description"] = clean_meta_str(row.get("description", ""))

    cache[urln] = {
        "url_normalized": urln,
        "title": row.get("title", ""),
        "description": row.get("description", ""),
        "http_status": row.get("http_status", ""),
        "fetch_error": row.get("fetch_error", ""),
    }
    return row


# ------------------ MAIN FILE ENRICHMENT ------------------ #

def enrich_file(
    in_path: Path,
    out_path: Path,
    cache_path: Optional[Path] = None,
) -> Path:
    """Read in_path, enrich, write out_path, maintain cache."""
    if cache_path is None:
        cache_path = out_path.with_name(out_path.stem + "_url_title_desc_cache.csv")

    cache = load_cache(cache_path)

    with open(in_path, "r", newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))

    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f_out:
            w = csv.DictWriter(f_out, fieldnames=["sourceurl", "title", "description", "http_status", "fetch_error"])
            w.writeheader()
        save_cache(cache_path, cache)
        print(f"Saved empty output: {out_path}")
        print(f"Cache: {cache_path}")
        print("Time: 0.00s  |  Rows: 0")
        return out_path

    extra_cols = ["title", "description", "http_status", "fetch_error"]
    fieldnames = list(rows[0].keys())
    for c in extra_cols:
        if c not in fieldnames:
            fieldnames.append(c)

    start = time.time()

    with requests.Session() as session:
        session.headers.update(HEADERS)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results = list(
                tqdm(
                    ex.map(lambda r: process_row(r, cache, session), rows),
                    total=len(rows),
                    desc=f"Enriching {in_path.name}",
                )
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    save_cache(cache_path, cache)

    elapsed = time.time() - start
    print(f"Saved: {out_path}")
    print(f"Cache: {cache_path}")
    print(f"Time: {elapsed:.2f}s  |  Rows: {len(rows):,}")

    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--cache", dest="cache_path", default=None)
    args = ap.parse_args()

    enrich_file(
        Path(args.in_path),
        Path(args.out_path),
        Path(args.cache_path) if args.cache_path else None,
    )