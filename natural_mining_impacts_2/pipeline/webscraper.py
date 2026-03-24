from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

# Optional fallback extractor
try:
    from newspaper import Article as _NPArticle
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = BASE_DIR / "data" / "interim" / "_state"
URLS_DIR = BASE_DIR / "data" / "urls"
SCRAPE_DIR = BASE_DIR / "data" / "processed" / "webscraped_daily"

# =========================
# Config
# =========================
SCRAPE_MAX_WORKERS = 8
SCRAPE_TIMEOUT = 20

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)


# ------------------ NORMALISATION ------------------ #

def normalize_url_basic(url: str) -> str:
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    return url


# ------------------ DATE EXTRACTION ------------------ #

def fallback_date_from_url(url: str) -> str:
    try:
        path = urlparse(url).path or ""
    except Exception:
        return "unknown"

    m = re.search(r"/(20\d{2})/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/", path)
    if m:
        yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"

    m = re.search(r"/afp/(\d{2})(\d{2})(\d{2})", path)
    if m:
        yy, mm, dd = m.group(1), m.group(2), m.group(3)
        return f"20{yy}-{mm}-{dd}"

    return "unknown"


def parse_date_str(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    try:
        dt = dtparser.parse(s, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return "unknown"


# ------------------ URL / HTTP HELPERS ------------------ #

def is_valid_http_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    url = url.strip()
    if not url:
        return False
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    try:
        p = urlparse(url)
        return bool(p.scheme in {"http", "https"} and p.netloc)
    except Exception:
        return False


def wayback_lookup(url: str, timeout: int = 15) -> dict:
    if not is_valid_http_url(url):
        return {
            "ok": False,
            "archive_url": "",
            "timestamp": "",
            "status": "",
            "error": "invalid_url",
        }

    api_url = "https://archive.org/wayback/available"
    try:
        resp = requests.get(
            api_url,
            params={"url": url},
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        snapshots = data.get("archived_snapshots", {}) or {}
        closest = snapshots.get("closest") or {}

        archive_url = closest.get("url", "") or ""
        timestamp = closest.get("timestamp", "") or ""
        status = str(closest.get("status", "") or "")

        if archive_url:
            return {
                "ok": True,
                "archive_url": archive_url,
                "timestamp": timestamp,
                "status": status,
                "error": "",
            }

        return {
            "ok": False,
            "archive_url": "",
            "timestamp": "",
            "status": "",
            "error": "no_snapshot_found",
        }

    except Exception as e:
        return {
            "ok": False,
            "archive_url": "",
            "timestamp": "",
            "status": "",
            "error": f"{type(e).__name__}: {e}",
        }


def fetch_html(url: str, timeout: int = 20) -> dict:
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        status_code = resp.status_code
        final_url = resp.url

        if status_code >= 400:
            return {
                "ok": False,
                "html": "",
                "status_code": status_code,
                "final_url": final_url,
                "error": f"HTTP {status_code}",
            }

        return {
            "ok": True,
            "html": resp.text or "",
            "status_code": status_code,
            "final_url": final_url,
            "error": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "html": "",
            "status_code": None,
            "final_url": url,
            "error": f"{type(e).__name__}: {e}",
        }


# ------------------ EXTRACTION ------------------ #

def extract_date_from_html(html_text: str) -> str:
    try:
        soup = BeautifulSoup(html_text or "", "html.parser")
    except Exception:
        return "unknown"

    meta_names = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "pubdate"}),
        ("meta", {"name": "publishdate"}),
        ("meta", {"name": "publish_date"}),
        ("meta", {"name": "date"}),
        ("meta", {"name": "Date"}),
        ("meta", {"name": "dc.date"}),
        ("meta", {"name": "DC.date"}),
        ("meta", {"property": "og:updated_time"}),
        ("meta", {"name": "parsely-pub-date"}),
    ]

    for tag, attrs in meta_names:
        el = soup.find(tag, attrs=attrs)
        if el and el.get("content"):
            d = parse_date_str(el.get("content"))
            if d != "unknown":
                return d

    t = soup.find("time")
    if t:
        if t.get("datetime"):
            d = parse_date_str(t.get("datetime"))
            if d != "unknown":
                return d
        d = parse_date_str(t.get_text(" ", strip=True))
        if d != "unknown":
            return d

    m = re.search(rf"({_MONTHS})\s+\d{{1,2}},\s+20\d{{2}}", soup.get_text(" ", strip=True))
    if m:
        return parse_date_str(m.group(0))

    return "unknown"


def extract_from_html(html_text: str, url: str = "") -> dict:
    try:
        text = trafilatura.extract(
            html_text,
            include_comments=False,
            include_tables=False,
        ) or ""
    except Exception:
        text = ""

    title = ""
    try:
        meta = trafilatura.metadata.extract_metadata(html_text)
        title = (meta.title or "") if meta else ""
    except Exception:
        title = ""

    if not title:
        try:
            soup = BeautifulSoup(html_text or "", "html.parser")
            if soup.title and soup.title.get_text(strip=True):
                title = soup.title.get_text(strip=True)
        except Exception:
            pass

    pub_date = extract_date_from_html(html_text)
    if pub_date == "unknown" and url:
        pub_date = fallback_date_from_url(url)

    return {
        "ok": bool(text.strip()),
        "text": text,
        "title": title,
        "html": html_text,
        "date": pub_date,
        "error": "" if text.strip() else "text_too_short_or_failed",
    }


def newspaper_extract(url: str, timeout: int = 20) -> dict:
    if not _HAS_NEWSPAPER:
        return {
            "ok": False,
            "text": "",
            "title": "",
            "html": "",
            "final_url": url,
            "error": "newspaper_not_installed",
        }

    try:
        art = _NPArticle(url)
        art.download()
        art.parse()
        text = art.text or ""
        title = art.title or ""
        return {
            "ok": bool(text.strip()),
            "text": text,
            "title": title,
            "html": "",
            "final_url": url,
            "error": "" if text.strip() else "newspaper_empty_text",
        }
    except Exception as e:
        return {
            "ok": False,
            "text": "",
            "title": "",
            "html": "",
            "final_url": url,
            "error": f"{type(e).__name__}: {e}",
        }


def scrape_article_with_wayback(url: str, timeout: int = 20, try_wayback: bool = True) -> dict:
    result = {
        "url_normalized": normalize_url_basic(url),
        "sourceurl": url,
        "scrape_ok": False,
        "scrape_status": "",
        "scrape_error": "",
        "text": "",
        "final_url": url,
        "http_status": None,
        "used_wayback": False,
        "wayback_url": "",
        "wayback_timestamp": "",
    }

    if not is_valid_http_url(url):
        result["scrape_status"] = "invalid_url"
        result["scrape_error"] = "sourceurl_not_valid_http_url"
        return result

    live_html = fetch_html(url, timeout=timeout)
    result["http_status"] = live_html["status_code"]
    result["final_url"] = live_html["final_url"]

    if live_html["ok"] and live_html["html"]:
        extracted = extract_from_html(live_html["html"], url=live_html["final_url"])
        if extracted["ok"]:
            result["scrape_ok"] = True
            result["scrape_status"] = "live_ok_trafilatura"
            result["text"] = extracted["text"]
            return result

    np_res = newspaper_extract(url, timeout=timeout)
    if np_res.get("ok"):
        result["scrape_ok"] = True
        result["scrape_status"] = "live_ok_newspaper"
        result["text"] = np_res.get("text", "") or ""
        result["scrape_error"] = ""
        return result

    live_error = live_html["error"] or np_res.get("error", "") or "live_scrape_failed"

    if try_wayback:
        wb = wayback_lookup(url, timeout=timeout)
        if wb["ok"]:
            result["used_wayback"] = True
            result["wayback_url"] = wb["archive_url"]
            result["wayback_timestamp"] = wb["timestamp"]

            time.sleep(0.2)

            wb_html = fetch_html(wb["archive_url"], timeout=timeout)
            if wb_html["ok"] and wb_html["html"]:
                extracted = extract_from_html(wb_html["html"], url=url)
                if extracted["ok"]:
                    result["scrape_ok"] = True
                    result["scrape_status"] = "wayback_ok"
                    result["text"] = extracted["text"]
                    result["final_url"] = wb["archive_url"]
                    result["scrape_error"] = ""
                    return result

            result["scrape_status"] = "wayback_failed"
            result["scrape_error"] = wb_html["error"] or "wayback_extraction_failed"
            return result

        result["scrape_status"] = "live_failed_no_wayback"
        result["scrape_error"] = f"{live_error} | wayback: {wb['error']}"
        return result

    result["scrape_status"] = "live_failed"
    result["scrape_error"] = live_error
    return result


# ------------------ BATCH SCRAPING ------------------ #

def scrape_many(urls: list[str], max_workers: int = SCRAPE_MAX_WORKERS, timeout: int = SCRAPE_TIMEOUT) -> pd.DataFrame:
    rows = []
    urls = [u for u in urls if isinstance(u, str) and u.strip()]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_url = {
            ex.submit(scrape_article_with_wayback, url, timeout, True): url
            for url in urls
        }

        for i, fut in enumerate(as_completed(fut_to_url), start=1):
            url = fut_to_url[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "url_normalized": normalize_url_basic(url),
                    "sourceurl": url,
                    "scrape_ok": False,
                    "scrape_status": "future_exception",
                    "scrape_error": f"{type(e).__name__}: {e}",
                    "text": "",
                    "final_url": url,
                    "http_status": None,
                    "used_wayback": False,
                    "wayback_url": "",
                    "wayback_timestamp": "",
                }
            rows.append(res)

            if i % 25 == 0 or i == len(fut_to_url):
                print(f"Scraped {i}/{len(fut_to_url)}")

    return pd.DataFrame(rows)


# ------------------ MAIN PIPELINE ENTRY ------------------ #

def main(target_date: str, force: bool = False) -> None:
    year, month, day = target_date[:4], target_date[4:6], target_date[6:8]

    out_dir = SCRAPE_DIR / year / month / day
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{target_date}_webscraped.csv"
    state_csv = STATE_DIR / f"webscrape_cache_{target_date}.csv"

    if out_csv.exists() and not force:
        print(f"Skipping webscraper for {target_date}: already done ({out_csv})")
        return

    in_csv = URLS_DIR / f"{target_date}.csv"
    if not in_csv.exists():
        print(f"Skipping webscraper: input not found at {in_csv}")
        return

    df = pd.read_csv(in_csv, encoding="utf-8", engine="python")
    if "url_normalized" not in df.columns:
        raise ValueError(f"Expected 'url_normalized' column in {in_csv}")

    urls = (
        df["url_normalized"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )

    print(f"\n--- Webscraping {len(urls)} URLs for {target_date} ---")
    scraped = scrape_many(urls)

    # Merge scrape results back onto original relevant_urls file
    merged = df.merge(scraped, on="url_normalized", how="left")

    merged.to_csv(out_csv, index=False)
    merged.to_csv(state_csv, index=False)

    ok_count = int(merged["scrape_ok"].fillna(False).sum()) if "scrape_ok" in merged.columns else 0
    fail_count = len(merged) - ok_count

    print(f"Success! Webscraper output saved.")
    print(f"Rows: {len(merged)} | Scrape ok: {ok_count} | Scrape failed: {fail_count}")
    print(f"Output CSV: {out_csv}\n")


if __name__ == "__main__":
    target_date = input("Enter date to scrape (YYYYMMDD): ").strip()
    main(target_date, force=False)