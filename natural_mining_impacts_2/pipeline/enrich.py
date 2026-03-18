from __future__ import annotations

import asyncio
import csv
import random
import re
import sys
import time
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm


# =========================
# CONFIG
# =========================


BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "interim" / "gdelt_event_context_daily"
OUTPUT_SUFFIX = "_enriched.csv"

USER_AGENT = "Mozilla/5.0 (compatible; LithiumQRA/1.0)"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TIMEOUT_S = 10
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.6

MAX_CONCURRENT = 200        # aiohttp handles this cheaply; tune up/down as needed
SLEEP_BETWEEN_REQ = (0.0, 0.03)

MAX_TITLE_CHARS = 300
MAX_DESC_CHARS = 800

CACHE_QUEUE_MAXSIZE = 50_000
STREAM_HEAD_MAX_BYTES = 256 * 1024   # stop reading after </head> or 256 KB

NO_RETRY_STATUSES = {400, 401, 403, 404, 405, 410, 451}


# =========================
# HELPERS
# =========================

TRACKING_PARAMS = {
    "gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "spm", "ref", "ref_src"
}

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    try:
        p = urlparse(url)
        query = [
            (k, v)
            for (k, v) in parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
        ]
        scheme = p.scheme or "http"
        netloc = (p.netloc or "").lower()
        path = p.path or ""
        return urlunparse((scheme, netloc, path, p.params, urlencode(query, doseq=True), ""))
    except Exception:
        return url

def truncate(s: Optional[str], n: int) -> str:
    s = (s or "").strip()
    return s[:n] if len(s) > n else s

def looks_like_xml(content_type: str, head_bytes: bytes) -> bool:
    ct = content_type.lower()
    if "xml" in ct or "rss" in ct or "atom" in ct:
        return True
    head = head_bytes[:300].lstrip().lower()
    return (
        head.startswith(b"<?xml")
        or head.startswith(b"<rss")
        or head.startswith(b"<feed")
        or head.startswith(b"<sitemapindex")
        or head.startswith(b"<urlset")
    )

def safe_get_meta_description(soup: BeautifulSoup) -> str:
    tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if tag and tag.get("content"):
        return tag.get("content", "")
    tag = soup.find("meta", attrs={"property": re.compile(r"^og:description$", re.I)})
    if tag and tag.get("content"):
        return tag.get("content", "")
    tag = soup.find("meta", attrs={"name": re.compile(r"^twitter:description$", re.I)})
    if tag and tag.get("content"):
        return tag.get("content", "")
    return ""

def read_csv_rows(path: Path) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv_rows(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# =========================
# CACHE WRITER THREAD
# (same single-writer pattern — runs in its own thread alongside the event loop)
# =========================

CACHE_FIELDS = ["url_normalized", "title", "meta_description", "http_status", "fetch_error"]

class CacheWriter:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.queue: Queue[dict] = Queue(maxsize=CACHE_QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.queue.put({"__stop__": "1"})
        self._thread.join(timeout=10)

    def submit(self, row: dict) -> None:
        self.queue.put(row)

    def _run(self) -> None:
        write_header = not self.cache_path.exists()
        f = open(self.cache_path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        if write_header:
            w.writeheader()
            f.flush()
        try:
            while not self._stop.is_set():
                try:
                    item = self.queue.get(timeout=0.5)
                except Empty:
                    continue
                if item.get("__stop__") == "1":
                    break
                w.writerow({k: item.get(k, "") for k in CACHE_FIELDS})
                if random.random() < 0.02:
                    f.flush()
        finally:
            f.flush()
            f.close()


# =========================
# CACHE I/O
# =========================

def load_cache(cache_path: Path) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u = (r.get("url_normalized") or "").strip()
            if u:
                cache[u] = r
    return cache

def load_existing_progress(out_path: Path) -> Dict[str, dict]:
    existing: Dict[str, dict] = {}
    if not out_path.exists():
        return existing
    with open(out_path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u = (r.get("url_normalized") or "").strip()
            if u:
                existing[u] = r
    return existing


# =========================
# ASYNC FETCH
# =========================

async def _read_head_bytes(response: aiohttp.ClientResponse) -> bytes:
    """Stream response chunks, stop once </head> is seen or cap is hit."""
    buf = b""
    needle = b"</head>"
    async for chunk in response.content.iter_chunked(8192):
        buf += chunk
        if needle in buf.lower():
            break
        if len(buf) >= STREAM_HEAD_MAX_BYTES:
            break
    return buf

async def fetch_title_meta(
    url: str,
    session: aiohttp.ClientSession,
) -> Tuple[str, str, int, str]:
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.get(url, timeout=timeout, allow_redirects=True) as resp:
                status = resp.status

                if status != 200:
                    if status in NO_RETRY_STATUSES:
                        return "", "", status, f"bad_status:{status}"
                    if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                        await asyncio.sleep(BACKOFF_BASE_S * (attempt + 1))
                        continue
                    return "", "", status, f"bad_status:{status}"

                raw = await _read_head_bytes(resp)
                if not raw:
                    return "", "", status, "empty_response"

                ct = resp.headers.get("Content-Type", "")
                parser = "xml" if looks_like_xml(ct, raw) else "lxml"

                encoding = resp.charset or "utf-8"
                try:
                    text = raw.decode(encoding, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    text = raw.decode("utf-8", errors="replace")

                soup = BeautifulSoup(text, parser)

                title = ""
                if soup.title and soup.title.string:
                    title = soup.title.string

                desc = safe_get_meta_description(soup)
                return truncate(title, MAX_TITLE_CHARS), truncate(desc, MAX_DESC_CHARS), status, ""

        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_BASE_S * (attempt + 1))
                continue
            return "", "", 0, "exception:TimeoutError"
        except Exception as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_BASE_S * (attempt + 1))
                continue
            return "", "", 0, f"exception:{type(e).__name__}"

    return "", "", 0, "failed_after_retries"


# =========================
# ASYNC PROCESSING
# =========================

async def process_row_async(
    row: dict,
    cache: Dict[str, dict],
    existing: Dict[str, dict],
    cache_writer: CacheWriter,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> dict:
    url = (row.get("sourceurl") or "").strip()
    url_norm = normalize_url(url)
    row["url_normalized"] = url_norm

    # Resume from prior successful run
    if url_norm in existing and existing[url_norm].get("http_status") == "200":
        return existing[url_norm]

    # Non-HTTP
    if not url_norm.startswith("http"):
        row.update({"title": "", "meta_description": "", "http_status": "", "fetch_error": "non_http"})
        return row

    # In-memory cache hit
    if url_norm in cache:
        c = cache[url_norm]
        row.update({
            "title": c.get("title", ""),
            "meta_description": c.get("meta_description", ""),
            "http_status": str(c.get("http_status", "")),
            "fetch_error": c.get("fetch_error", ""),
        })
        return row

    # Small jitter to avoid thundering-herd on a single domain
    await asyncio.sleep(random.uniform(*SLEEP_BETWEEN_REQ))

    async with semaphore:
        title, desc, status, err = await fetch_title_meta(url_norm, session)

    row.update({"title": title, "meta_description": desc, "http_status": str(status), "fetch_error": err})

    cache_writer.submit({
        "url_normalized": url_norm,
        "title": title,
        "meta_description": desc,
        "http_status": str(status),
        "fetch_error": err,
    })
    return row


async def enrich_file_async(
    in_path: Path,
    cache: Dict[str, dict],
    cache_writer: CacheWriter,
) -> None:
    out_path = in_path.with_name(in_path.stem.replace("_filtered", "") + OUTPUT_SUFFIX)

    if out_path.exists():
        print(f"Skipping (already enriched): {out_path.name}")
        return

    rows = read_csv_rows(in_path)
    if not rows:
        return

    existing = load_existing_progress(out_path)

    fieldnames = list(rows[0].keys())
    for extra in ["url_normalized", "title", "meta_description", "http_status", "fetch_error"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, ssl=False)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)

    start = time.time()
    results: List[dict] = [None] * len(rows)

    async with aiohttp.ClientSession(
        headers=HEADERS,
        connector=connector,
        timeout=timeout,
    ) as session:
        tasks = [
            process_row_async(rows[i], cache, existing, cache_writer, session, semaphore)
            for i in range(len(rows))
        ]

        pbar = tqdm(total=len(tasks), desc=f"Enriching {in_path.parent.name}/{in_path.name}")
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            # as_completed doesn't preserve order so we track by URL after
            result = await coro
            results[i] = result
            pbar.update(1)
        pbar.close()

    # Restore stable ordering by url_normalized → original index
    url_to_idx = {normalize_url((rows[i].get("sourceurl") or "").strip()): i
                  for i in range(len(rows))}
    ordered: List[dict] = [None] * len(rows)
    used = set()
    for r in results:
        u = (r or {}).get("url_normalized", "")
        idx = url_to_idx.get(u)
        if idx is not None and idx not in used:
            ordered[idx] = r
            used.add(idx)
        else:
            # fallback: fill next empty slot
            for j in range(len(ordered)):
                if ordered[j] is None:
                    ordered[j] = r
                    break

    write_csv_rows(out_path, fieldnames, ordered)

    elapsed = time.time() - start
    rps = (len(rows) / elapsed) if elapsed > 0 else 0.0
    print(f"\n--- ENRICH REPORT: {in_path.name} ---")
    print(f"Rows: {len(rows)} | Time: {elapsed:.1f}s | Throughput: {rps:.2f} rows/s")
    print(f"Saved: {out_path}\n")


async def _run_all(files: List[Path], cache: Dict[str, dict], cache_writer: CacheWriter) -> None:
    """Process all date files concurrently."""
    await asyncio.gather(*[enrich_file_async(fp, cache, cache_writer) for fp in files])


# =========================
# ENTRY POINT
# =========================

def main(target_date: str) -> None:
    # cache_path = Path(f"data/interim/_state/url_title_meta_cache_{target_date}.csv")
    cache_path = Path(__file__).resolve().parent.parent / "data" / "interim" / "_state" / f"url_title_meta_cache_{target_date}.csv"
    cache = load_cache(cache_path)

    files = list(BASE_DIR.rglob(f"*{target_date}*_deduped_filtered.csv"))
    if not files:
        print(f"No filtered files found for {target_date} under {BASE_DIR}")
        return

    cache_writer = CacheWriter(cache_path)
    cache_writer.start()

    try:
        asyncio.run(_run_all(files, cache, cache_writer))
    finally:
        cache_writer.stop()


if __name__ == "__main__":
    day = input("Enter date (YYYYMMDD): ").strip()
    if not day:
        print("No date provided.")
        sys.exit(1)
    main(day)