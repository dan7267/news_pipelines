"""second_classifier.py

Stage 4: scrape + LLM classify mining-relatedness and impact taxonomy.

This is a direct refactor of second_classifier.ipynb, with paths parameterised.

Input CSV must include at least:
- sourceurl
- title (optional)
- description (optional)

Output is intended to be EXACTLY the same as the final CSV produced by
second_classifier.ipynb (same added columns and formatting), assuming
same model, same prompts, and same network responses.

Notes:
- Requires OPENAI_API_KEY in environment or .env.
- Writes/uses a scrape cache CSV alongside the output.
"""

from __future__ import annotations

# --- Notebook cell 0 (date + scrape helpers) ---
import requests
import trafilatura
import json
import re
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from urllib.parse import urlparse, quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Try to import Newspaper3k as an optional fallback extractor
try:
    from newspaper import Article as _NPArticle
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False


# ------------------ DATE EXTRACTION ------------------ #

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)

LLM_MAX_WORKERS = 8


def fallback_date_from_url(url: str) -> str:
    """
    Best-effort fallback date extraction from URL patterns.
    Returns ISO date 'YYYY-MM-DD' or 'unknown'.

    Supports:
      - /YYYY/MM/DD/  (common on WordPress)
      - /afp/YYMMDD... (AFP wire URLs)
    """
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
    """
    Return the closest available Wayback snapshot for a URL.

    Output:
      {
        "ok": bool,
        "archive_url": str,
        "timestamp": str,
        "status": str,
        "error": str,
      }
    """
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
    
def extract_from_html(html_text: str, url: str = "") -> dict:
    """
    Extract article text/title/date from raw HTML.
    """
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

def extract_date_from_html(html_text: str) -> str:
    """Try common meta tags and time elements."""

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

    # <time datetime="...">
    t = soup.find("time")
    if t:
        if t.get("datetime"):
            d = parse_date_str(t.get("datetime"))
            if d != "unknown":
                return d
        # sometimes time tag contains the date
        d = parse_date_str(t.get_text(" ", strip=True))
        if d != "unknown":
            return d

    # Regex fallback (e.g. "December 8, 2025")
    m = re.search(rf"({ _MONTHS })\s+\d{{1,2}},\s+20\d{{2}}", soup.get_text(" ", strip=True))
    if m:
        return parse_date_str(m.group(0))

    return "unknown"


def trafilatura_extract(url: str, timeout: int = 20) -> dict:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"ok": False, "text": "", "title": "", "html": "", "final_url": url, "error": "fetch_url_returned_empty"}

        text = trafilatura.extract(downloaded) or ""
        title = ""
        try:
            meta = trafilatura.metadata.extract_metadata(downloaded)
            title = (meta.title or "") if meta else ""
        except Exception:
            title = ""

        return {"ok": bool(text.strip()), "text": text, "title": title, "html": downloaded, "final_url": url, "error": ""}
    except Exception as e:
        return {"ok": False, "text": "", "title": "", "html": "", "final_url": url, "error": f"{type(e).__name__}: {e}"}

def newspaper_extract(url: str, timeout: int = 20) -> dict:
    """Fallback: Newspaper3k extraction."""
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
    """
    Try:
      1. live trafilatura
      2. live newspaper3k
      3. Wayback snapshot fetch + extract

    Always returns a fixed schema.
    """
    result = {
        "sourceurl": url,
        "scrape_success": False,
        "scrape_status": "",
        "scrape_error": "",
        "scraped_text": "",
        "scraped_title": "",
        "scraped_date": "unknown",
        "final_url": url,
        "http_status": None,
        "used_wayback": False,
        "wayback_url": "",
        "wayback_timestamp": "",
    }

    # ---------- Validate ----------
    if not is_valid_http_url(url):
        result["scrape_status"] = "invalid_url"
        result["scrape_error"] = "sourceurl_not_valid_http_url"
        return result

    # ---------- Try live trafilatura ----------
    live_html = fetch_html(url, timeout=timeout)
    result["http_status"] = live_html["status_code"]
    result["final_url"] = live_html["final_url"]

    if live_html["ok"] and live_html["html"]:
        extracted = extract_from_html(live_html["html"], url=live_html["final_url"])
        if extracted["ok"]:
            result["scrape_success"] = True
            result["scrape_status"] = "live_ok_trafilatura"
            result["scraped_text"] = extracted["text"]
            result["scraped_title"] = extracted["title"]
            result["scraped_date"] = extracted["date"]
            return result

    # ---------- Try live newspaper3k ----------
    np_res = newspaper_extract(url, timeout=timeout)
    if np_res.get("ok"):
        result["scrape_success"] = True
        result["scrape_status"] = "live_ok_newspaper"
        result["scraped_text"] = np_res.get("text", "") or ""
        result["scraped_title"] = np_res.get("title", "") or ""
        result["scraped_date"] = fallback_date_from_url(url)
        result["scrape_error"] = ""
        return result

    live_error = live_html["error"] or np_res.get("error", "") or "live_scrape_failed"

    # ---------- Wayback fallback ----------
    if try_wayback:
        wb = wayback_lookup(url, timeout=timeout)
        if wb["ok"]:
            result["used_wayback"] = True
            result["wayback_url"] = wb["archive_url"]
            result["wayback_timestamp"] = wb["timestamp"]

            # Small pause is polite and can reduce transient issues
            time.sleep(0.2)

            wb_html = fetch_html(wb["archive_url"], timeout=timeout)
            if wb_html["ok"] and wb_html["html"]:
                extracted = extract_from_html(wb_html["html"], url=url)
                if extracted["ok"]:
                    result["scrape_success"] = True
                    result["scrape_status"] = "wayback_ok"
                    result["scraped_text"] = extracted["text"]
                    result["scraped_title"] = extracted["title"]
                    result["scraped_date"] = extracted["date"]
                    result["final_url"] = wb["archive_url"]
                    result["scrape_error"] = ""
                    return result

            result["scrape_status"] = "wayback_failed"
            result["scrape_error"] = wb_html["error"] or "wayback_extraction_failed"
            return result

        result["scrape_status"] = "live_failed_no_wayback"
        result["scrape_error"] = f"{live_error} | wayback: {wb['error']}"
        return result

    # ---------- No wayback attempted ----------
    result["scrape_status"] = "live_failed"
    result["scrape_error"] = live_error
    return result



IMPACT_TAXONOMY = """
Environmental
- Land, Soil & Geology
    - Overburden removal and waste rock generation
    - Tailings generation and management
    - Topsoil degradation and loss of soil fertility
    - Deep soil compaction and subsurface degradation
    - Land disturbance and land-use change
    - Landscape and topography alteration
    - Surface ground instability and subsidence
    - Slope instability and landslides
    - Fly rock from blasting
    - Ground vibration and air overpressure (blasting impacts)
    - Transport corridors and access road impacts
    - Geothermal effects at depth (elevated subsurface temperatures)

- Water Resources
    - Acid mine drainage
    - Surface water pollution and quality degradation
    - Groundwater contamination
    - Freshwater ecotoxicity (heavy metals, pH changes, sensitive species)
    - Eutrophication of water bodies
    - Water table alteration and drainage issues
    - High water consumption
    - Water management challenges

- Air & Climate
    - Air quality degradation (dust, diesel exhaust, blasting emissions)
    - Particulate matter emissions
    - Dust generation
    - Photochemical ozone formation (smog)
    - Ozone depletion (emissions from equipment/explosives)
    - Greenhouse gas emissions (GHGs)
    - Local atmospheric heating and microclimate change
    - Temperature inversions trapping pollution
    - Fossil fuel consumption
    - Energy demand and consumption
    - Clean/renewable energy integration (mitigation-related impact)
    - Carbon sink destruction (deforestation, land clearing)

- Biodiversity & Ecosystems
    - Ecosystem degradation and habitat loss/fragmentation
    - Deforestation
    - Impacts on terrestrial ecotoxicity
    - Impacts on aquatic ecosystems and life below water
    - Loss of ecosystem services
    - Ecosystem recovery through reclamation and restoration (post-mining impact)

- Noise & Disturbance
    - Noise pollution (drilling, blasting, crushing, hauling)


Social
- Health, Safety & Well-being
    - Worker health and safety risks
    - Community exposure to dust, noise, pollution
    - Accidents and fatalities (including structural failures such as tailings dams)
    - Equipment safety risks
    - Material safety risks
    - Mental and physical health impacts on communities
    - Public safety concerns from blasting, vibrations, fly rock

- Livelihoods & Economy
    - Employment creation and job security
    - Unemployment risks post-closure
    - Income generation and poverty reduction or exacerbation
    - Business opportunities (local supply chains, services)
    - Human capital development (skills, training, workforce development)
    - Equipment and materials availability affecting local labor markets

- Living Conditions & Social Fabric
    - Livability (cost of living, food security, communication services)
    - Social relationships and family/community cohesion
    - Demographic changes (migration, influx of workers, displacement)
    - Social infrastructure and amenities (education, healthcare, public services, leisure)
    - Education and training opportunities
    - Tourism attraction or decline
    - Cultural impacts (tangible and intangible heritage, traditions, identity)

- Equity, Rights & Vulnerable Groups
    - Freedom and justice impacts in nearby communities
    - Stakeholder inclusion and community participation
    - Future generations’ rights and intergenerational equity
    - Land ownership impacts and regional value changes
    - Indigenous peoples’ rights and conflicts with mining
    - Child labor and forced labor risks
    - Crime and social disorder (corruption, violence, substance abuse)
    - Wealth distribution and inequality
    
- Quality of Work Life
    - Job satisfaction
    - Labor conditions and workplace dignity


Governance
- Regulation, Compliance & Institutions
    - Legislative and regulatory framework effectiveness
    - Enforcement of environmental and social regulations
    - Mining permitting and licensing practices 
    - Compliance with national and international standards 

- Transparency & Accountability
    - Corporate transparency and reporting
    - Monitoring and disclosure of environmental and social performance 
    - Anti-corruption and bribery risks
    - Accountability for accidents and environmental damage

- Stakeholder Engagement & Rights
    - Stakeholder inclusion in decision-making
    - Community consent and consultation (FPIC – Free, Prior and Informed Consent)
    - Conflict management between companies and communities
    - Protection of indigenous land and resource rights

- Risk Management & Long-Term Stewardship
    - Tailings and waste governance
    - Closure planning and post-mining reclamation governance
    - Long-term liability management (pollution, water treatment, land stability)
    - Intergenerational responsibility and resource depletion ethics

- Reputation & Social License
    - Mining image and public perception
    - Social license to operate
    - Trust between mining companies, government, and communities
"""









# --- Notebook cell 1 (stage2 implementation), with path params ---

# stage2_mining_impacts_with_scrape_levels.py
# (refactored from notebook)

from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import os
import time
import random
import csv

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MODEL = "gpt-5-mini"

# scraping
SCRAPE_TIMEOUT_S = 20
MIN_TEXT_CHARS = 300
MAX_WORKERS = 10
SLEEP_BETWEEN_REQ = (0.05, 0.15)

SCRAPE_CACHE_FIELDS = [
    "url_normalized",
    "final_url",
    "scrape_ok",
    "scrape_status",
    "scrape_error",
    "scraped_title",
    "scraped_published_date",
    "text",
    "http_status",
    "used_wayback",
    "wayback_url",
    "wayback_timestamp",
]

# prompt sizing
MAX_TEXT_CHARS_FOR_LLM = 14000


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()

MIN_METADATA_FALLBACK_CHARS = 30

def build_metadata_fallback_text(row: Dict[str, Any]) -> str:
    title = _norm_str(row.get("title"))
    description = _norm_str(row.get("description"))

    parts = []
    if title:
        parts.append(f"TITLE: {title}")
    if description:
        parts.append(f"DESCRIPTION: {description}")

    return "\n".join(parts).strip()


def normalize_url_basic(url: str) -> str:
    """Light normalization to make scrape cache more stable."""
    try:
        p = urlparse((url or "").strip())
        return f"{p.scheme}://{p.netloc}{p.path}" if p.scheme and p.netloc else (url or "").strip()
    except Exception:
        return (url or "").strip()


def load_scrape_cache(cache_path: Path) -> Dict[str, Dict[str, str]]:
    cache: Dict[str, Dict[str, str]] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row.get("url_normalized") or "").strip()
            if k:
                cache[k] = {c: row.get(c, "") for c in SCRAPE_CACHE_FIELDS}
    return cache


def save_scrape_cache(cache_path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCRAPE_CACHE_FIELDS)
        w.writeheader()
        for _, row in cache.items():
            w.writerow({c: row.get(c, "") for c in SCRAPE_CACHE_FIELDS})


def scrape_one(url: str) -> Dict[str, Any]:
    """
    Scrape text + title + published date (best effort),
    with Wayback fallback.
    """

    url = (url or "").strip()
    url_norm = normalize_url_basic(url)

    # politeness
    time.sleep(random.uniform(*SLEEP_BETWEEN_REQ))

    res = scrape_article_with_wayback(
        url=url,
        timeout=SCRAPE_TIMEOUT_S,
        try_wayback=True,
    )

    text = (res.get("scraped_text") or "").strip()
    title = (res.get("scraped_title") or "").strip()
    pub = _norm_str(res.get("scraped_date")) or "unknown"

    ok = bool(res.get("scrape_success")) and len(text) >= MIN_TEXT_CHARS

    if not ok:
        return {
            "url_normalized": url_norm,
            "final_url": res.get("final_url", url),
            "scrape_ok": False,
            "scrape_status": res.get("scrape_status", "text_too_short_or_failed"),
            "scrape_error": res.get("scrape_error", ""),
            "scraped_title": title,
            "scraped_published_date": pub if pub != "unknown" else fallback_date_from_url(url),
            "text": "",
            "http_status": res.get("http_status"),
            "used_wayback": bool(res.get("used_wayback", False)),
            "wayback_url": res.get("wayback_url", ""),
            "wayback_timestamp": res.get("wayback_timestamp", ""),
        }

    return {
        "url_normalized": url_norm,
        "final_url": res.get("final_url", url),
        "scrape_ok": True,
        "scrape_status": res.get("scrape_status", "ok"),
        "scrape_error": res.get("scrape_error", ""),
        "scraped_title": title,
        "scraped_published_date": pub,
        "text": text,
        "http_status": res.get("http_status"),
        "used_wayback": bool(res.get("used_wayback", False)),
        "wayback_url": res.get("wayback_url", ""),
        "wayback_timestamp": res.get("wayback_timestamp", ""),
    }


def llm_stage2(
    client: OpenAI,
    url: str,
    scraped_title: str,
    published_date: str,
    article_text: str,
    input_mode: str = "scraped_text",
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> Dict[str, Any]:
    """LLM classifier + impact extractor (multi impacts)."""

    # Truncate to keep within prompt budget
    text = (article_text or "")[:MAX_TEXT_CHARS_FOR_LLM]

    system = f"""You are a strict information extraction system for Zambia mining news.

You will be given a news article's text.

Tasks:
1) Decide if the article is clearly about mining/minerals/metals/extraction/processing or related mining supply chains.
   - mining_related: true/false (true only if mining/minerals are clearly present)
   - mining_related_confidence: 0 to 1

2) Determine if the article is written about Zambia or the specific event is in Zambia.
   - in_zambia: true/false
   - in_zambia_confidence: 0 to 1

3) IF AND ONLY IF mining_related = true AND in_zambia = true, extract any explicitly supported impacts from mining and categorise each into 3 levels using ONLY the taxonomy below:

TAXONOMY (YOU MUST USE THESE EXACT LABELS):
{IMPACT_TAXONOMY}
   
Rules:
- impacts must be a list (possible empty).
- Each impact MUST be supported by a direct quote from the article text. If you cannot quote supporting text, DO NOT include the impact.

- For each impact:
    - level1 MUST be exactly one of: Environmental, Social, Governance
    - level2 MUST be one of the subcategories under that level1, with exact wording.
    - level3 MUST be one of the specific impacts under that level2, with exact wording

- You MUST select level2 and level3 by COPYING the exact text from the taxonomy. Do NOT paraphrase, do NOT invent new labels. 
- Do not force impacts into a category if not clearly supported by the text. It is okay to have zero impacts. 

- Only include impacts explicitly support by the article text
- Only include impacts clearly linked to mining activity in Zambia
- impact_evidence must be a list of objects with level1, level2, level3 and snippets (text excerpts from the text that support the impact classification)
- If scrape failed or text is too short, mining_related must be false
- Return JSON only with keys:
  in_zambia, in_zambia_confidence, mining_related, mining_related_confidence, impacts, impact_confidence, impact_evidence, mine_name, region, mineral_type, mining_company
Input mode:
- If input_mode = "scraped_text", the text is article body text.
- If input_mode = "metadata_fallback", the text is only metadata-derived text (title/description), not the full article.

Extra rules for metadata_fallback:
- You may still classify mining_related and in_zambia if clearly supported.
- Be more conservative than usual.
- Do NOT extract impacts unless they are explicitly and unambiguously supported by the metadata text.
- If impacts are not clearly supported, return an empty impacts list.
  

ADDITIONAL ENTITY EXTRACTION

From the article text, extract the following mining-related entities

Rules:
- Only extract information that appears clearly in the article.
- Do not infer missing information.
- If a field is not mentioned, return null.
- If mining_related is false or in_zambia is false, set mine_name, mining_company, region and mineral_type to null.

Fields:

mineral_type
The main mineral or commodity associated with the mine or disruption.
Examples: copper, cobalt, lithium, nickel, gold, iron ore, coal, rare earths.

region
The subnational region associated with the mine/event in Zambia, such as province, district, town or nearby area.
Examples: Copperbelt Province, North-Western Province, Solwezi, Kitwe, Chingola.

mine_name
The specific mine or mining project if mentioned.

mining_company
The company operating or owning the mine. Return the company name only (no descriptors). 
"""

    user = {
        "url": url,
        "scraped_title": scraped_title,
        "published_date": published_date,
        "input_mode": input_mode,
        "text": text,
    }

    max_retries = 3
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                timeout=timeout,
                response_format={"type": "json_object"},
            )

            content = resp.choices[0].message.content or "{}"
            return json.loads(content)

        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                sleep_s = 2 ** attempt
                print(f"LLM call failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}. Retrying in {sleep_s}s...")
                time.sleep(sleep_s)
            else:
                print(f"LLM call failed after {max_retries} attempts: {type(e).__name__}: {e}")

    # Conservative fallback after retries exhausted
    return {
        "in_zambia": False,
        "in_zambia_confidence": 0.0,
        "mining_related": False,
        "mining_related_confidence": 0.0,
        "impacts": [],
        "impact_confidence": 0.0,
        "impact_evidence": [],
        "llm_error": f"{type(last_err).__name__}: {last_err}" if last_err else "unknown_error",
        "mineral_type": None,
        "region": None,
        "mine_name": None,
        "mining_company": None,
    }


def run_stage2(
    in_path: Path,
    out_path: Path,
    model: str = DEFAULT_MODEL,
    max_rows: Optional[int] = None,
    scrape_cache_path: Optional[Path] = None,
) -> None:
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if scrape_cache_path is None:
        scrape_cache_path = out_path.with_name(out_path.stem + "_scrape_cache.csv")

    df = pd.read_csv(in_path)

    if max_rows:
        df = df.head(max_rows).copy()

    # Ensure required input column
    if "sourceurl" not in df.columns:
        raise ValueError("Expected a 'sourceurl' column.")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found.")

    client = OpenAI(api_key=api_key)

    rows = df.to_dict(orient="records")
    print(f"Stage2: scraping + LLM on {len(rows):,} rows...")

    # --- scrape with cache ---
    scrape_cache = load_scrape_cache(scrape_cache_path)

    def scrape_cached(url: str) -> Dict[str, Any]:
        k = normalize_url_basic(url)
        if k in scrape_cache:
            cached = scrape_cache[k]
            return {
                "url_normalized": k,
                "final_url": cached.get("final_url", url),
                "scrape_ok": str(cached.get("scrape_ok", "")).lower() in {"true", "1", "yes"},
                "scrape_status": cached.get("scrape_status", ""),
                "scrape_error": cached.get("scrape_error", ""),
                "scraped_title": cached.get("scraped_title", ""),
                "scraped_published_date": cached.get("scraped_published_date", "unknown"),
                "text": cached.get("text", ""),
                "http_status": cached.get("http_status", ""),
                "used_wayback": str(cached.get("used_wayback", "")).lower() in {"true", "1", "yes"},
                "wayback_url": cached.get("wayback_url", ""),
                "wayback_timestamp": cached.get("wayback_timestamp", ""),
            }

        try:
            out = scrape_one(url)
        except Exception as e:
            out = {
                "url_normalized": k,
                "final_url": url,
                "scrape_ok": False,
                "scrape_status": f"scrape_exception:{type(e).__name__}",
                "scrape_error": f"{type(e).__name__}: {e}",
                "scraped_title": "",
                "scraped_published_date": "unknown",
                "text": "",
                "http_status": "",
                "used_wayback": False,
                "wayback_url": "",
                "wayback_timestamp": "",
            }

        # save into cache dict
        scrape_cache[out["url_normalized"]] = {
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

    # Parallel scrape
    urls = [_norm_str(r.get("sourceurl")) for r in rows]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        scraped = list(tqdm(ex.map(scrape_cached, urls), total=len(urls), desc="Scraping"))

    save_scrape_cache(scrape_cache_path, scrape_cache)

    # --- LLM ---
    llm_jobs = []
    total_rows = len(rows)

    # First: attach scrape fields and handle non-LLM rows immediately
    for idx, (r, sc) in enumerate(zip(rows, scraped)):
        r["scrape_status"] = sc.get("scrape_status", "")
        r["scrape_error"] = sc.get("scrape_error", "")
        r["scraped_title"] = sc.get("scraped_title", "")
        r["scraped_published_date"] = sc.get("scraped_published_date", "unknown")
        r["final_url"] = sc.get("final_url", _norm_str(r.get("sourceurl")))
        r["http_status"] = sc.get("http_status", "")
        r["used_wayback"] = bool(sc.get("used_wayback", False))
        r["wayback_url"] = sc.get("wayback_url", "")
        r["wayback_timestamp"] = sc.get("wayback_timestamp", "")

        scraped_text = (sc.get("text") or "").strip()
        metadata_text = build_metadata_fallback_text(r)

        has_scraped_text = bool(sc.get("scrape_ok", False)) and len(scraped_text) >= MIN_TEXT_CHARS
        has_metadata_fallback = len(metadata_text) >= MIN_METADATA_FALLBACK_CHARS

        if has_scraped_text:
            llm_jobs.append((idx, r, sc, scraped_text, "scraped_text"))

        elif has_metadata_fallback:
            # Fallback: allow stage 2 to classify from title/description
            r["scrape_status"] = (r.get("scrape_status", "") or "") + "|metadata_fallback"
            llm_jobs.append((idx, r, sc, metadata_text, "metadata_fallback"))

        else:
            # No usable scrape text and no usable metadata
            r["mining_related"] = False
            r["mining_related_confidence"] = 0.0
            r["impact_level1"] = ""
            r["impact_level2"] = ""
            r["impact_level3"] = ""
            r["impact_confidence"] = 0.0
            r["impact_evidence"] = ""
            r["in_zambia"] = False
            r["in_zambia_confidence"] = 0.0
            r["mineral_type"] = ""
            r["region"] = ""
            r["mine_name"] = ""
            r["mining_company"] = ""

    print(f"Rows eligible for LLM: {len(llm_jobs):,}")

    def run_one_llm_job(job):
        idx, r, sc, text, input_mode = job

        llm_title = _norm_str(sc.get("scraped_title")) or _norm_str(r.get("title"))
        llm_date = _norm_str(sc.get("scraped_published_date"))

        result = llm_stage2(
            client=client,
            url=_norm_str(r.get("sourceurl")),
            scraped_title=llm_title,
            published_date=llm_date,
            article_text=text,
            model=model,
            input_mode=input_mode,
        )
        return idx, result, input_mode

    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as ex:
        futures = [ex.submit(run_one_llm_job, job) for job in llm_jobs]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM classify"):
            idx, result, input_mode = fut.result()
            r = rows[idx]
            r["stage2_input_mode"] = input_mode

            r["mining_related"] = bool(result.get("mining_related", False))
            r["mining_related_confidence"] = round(float(result.get("mining_related_confidence", 0.0)), 3)
            r["in_zambia"] = bool(result.get("in_zambia", False))
            r["in_zambia_confidence"] = round(float(result.get("in_zambia_confidence", 0.0)), 3)

            gate_ok = bool(r["in_zambia"]) and bool(r["mining_related"])
            r["impact_confidence"] = round(float(result.get("impact_confidence", 0.0)), 3) if gate_ok else 0.0

            r["mineral_type"] = _norm_str(result.get("mineral_type"))
            r["region"] = _norm_str(result.get("region"))
            r["mine_name"] = _norm_str(result.get("mine_name"))
            r["mining_company"] = _norm_str(result.get("mining_company"))

            impacts = result.get("impacts", [])
            if gate_ok and isinstance(impacts, list) and impacts:
                l1s, l2s, l3s = [], [], []
                for it in impacts:
                    if not isinstance(it, dict):
                        continue
                    l1s.append(_norm_str(it.get("level1")))
                    l2s.append(_norm_str(it.get("level2")))
                    l3s.append(_norm_str(it.get("level3")))
                r["impact_level1"] = " || ".join([x for x in l1s if x])
                r["impact_level2"] = " || ".join([x for x in l2s if x])
                r["impact_level3"] = " || ".join([x for x in l3s if x])
            else:
                r["impact_level1"] = ""
                r["impact_level2"] = ""
                r["impact_level3"] = ""

            ev_items = result.get("impact_evidence", [])
            flat_parts = []

            if gate_ok:
                if isinstance(ev_items, str):
                    ev_items = [ev_items]

                if isinstance(ev_items, list):
                    for item in ev_items:
                        if isinstance(item, str):
                            s = _norm_str(item)
                            if s:
                                flat_parts.append(f"UNLINKED -> “{s}”")
                            continue

                        if isinstance(item, dict):
                            l1 = _norm_str(item.get("level1"))
                            l2 = _norm_str(item.get("level2"))
                            l3 = _norm_str(item.get("level3"))
                            snips = item.get("snippets", [])

                            if isinstance(snips, str):
                                snips = [snips]

                            cleaned_snips = [f"“{_norm_str(s)}”" for s in snips if _norm_str(s)]
                            label = " | ".join([x for x in [l1, l2, l3] if x])

                            if label and cleaned_snips:
                                flat_parts.append(f"{label} -> {' | '.join(cleaned_snips)}")
                            elif label:
                                flat_parts.append(f"{label} ->")
                            elif cleaned_snips:
                                flat_parts.append(f"UNLINKED -> {' | '.join(cleaned_snips)}")

            r["impact_evidence"] = " || ".join(flat_parts)

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSaved: {out_path}")
    if "mining_related" in out_df.columns:
        print("Mining-related True:", int(out_df["mining_related"].sum()))
    any_impact = out_df["impact_level3"].fillna("").astype(str).str.strip() != ""
    print("With ≥1 impact:", int(any_impact.sum()))
    print(f"Scrape cache: {scrape_cache_path}")

def debug_test_url(url: str) -> int:
    """
    One-off URL test for debugging scraper/extractor.
    Returns process exit code: 0 if ok-ish, 1 otherwise.
    """
    url = (url or "").strip()
    if not url:
        print("No URL provided.")
        return 1

    print("\n=== TEST URL ===")
    print("URL:", url)
    print("Normalized:", normalize_url_basic(url))
    print(f"Settings: SCRAPE_TIMEOUT_S={SCRAPE_TIMEOUT_S}, MIN_TEXT_CHARS={MIN_TEXT_CHARS}\n")

    # Try trafilatura
    tri = trafilatura_extract(url, timeout=SCRAPE_TIMEOUT_S)
    tri_text = (tri.get("text") or "").strip()
    tri_title = (tri.get("title") or "").strip()
    tri_html = tri.get("html") or ""

    print("--- Trafilatura ---")
    print("ok:", bool(tri.get("ok")))
    print("final_url:", tri.get("final_url", url))
    print("html_len:", len(tri_html))
    print("title_len:", len(tri_title))
    print("text_len:", len(tri_text))

    print("error:", tri.get("error",""))
    if tri_title:
        print("title:", tri_title[:200])
    if tri_text:
        print("text_preview:", tri_text[:400].replace("\n", " ") + ("..." if len(tri_text) > 400 else ""))
    else:
        print("text_preview: <EMPTY>")

    pub_html = "unknown"
    if tri_html:
        pub_html = extract_date_from_html(tri_html)
    pub_url = fallback_date_from_url(url)

    print("published_date_from_html:", pub_html)
    print("published_date_from_url:", pub_url)

    # Try newspaper fallback
    print("\n--- Newspaper3k ---")
    if not _HAS_NEWSPAPER:
        print("newspaper3k not installed/available.")
        npd_text = ""
        npd_title = ""
    else:
        npd = newspaper_extract(url, timeout=SCRAPE_TIMEOUT_S)
        npd_text = (npd.get("text") or "").strip()
        npd_title = (npd.get("title") or "").strip()
        print("ok:", bool(npd.get("ok")))
        print("title_len:", len(npd_title))
        print("text_len:", len(npd_text))
        if npd_title:
            print("title:", npd_title[:200])
        if npd_text:
            print("text_preview:", npd_text[:400].replace("\n", " ") + ("..." if len(npd_text) > 400 else ""))
        else:
            print("text_preview: <EMPTY>")

    # Emulate scrape_one decision
    best_len = max(len(tri_text), len(npd_text))
    passes_threshold = best_len >= MIN_TEXT_CHARS

    print("\n--- Decision ---")
    print("best_extracted_len:", best_len)
    print("passes_MIN_TEXT_CHARS:", passes_threshold)

    # Exit code: 0 if it looks usable, else 1
    return 0 if passes_threshold else 1


###TEST BLOCK###

# if __name__ == "__main__":
#     import argparse
#     import sys

#     ap = argparse.ArgumentParser()
#     ap.add_argument("--in", dest="in_path", required=False)
#     ap.add_argument("--out", dest="out_path", required=False)
#     ap.add_argument("--model", dest="model", default=DEFAULT_MODEL)
#     ap.add_argument("--max-rows", dest="max_rows", type=int, default=None)
#     ap.add_argument("--scrape-cache", dest="scrape_cache", default=None)

#     # NEW: one-off scrape test mode
#     ap.add_argument("--test-url", dest="test_url", default=None,
#                     help="If set, runs a one-off scrape/extract debug on this URL and exits.")

#     args = ap.parse_args()

#     # Test mode: run and exit
#     if args.test_url:
#         code = debug_test_url(args.test_url)
#         sys.exit(code)

#     # Normal pipeline mode
#     if not args.in_path or not args.out_path:
#         ap.error("--in and --out are required unless --test-url is used.")

#     run_stage2(
#         Path(args.in_path),
#         Path(args.out_path),
#         model=args.model,
#         max_rows=args.max_rows,
#         scrape_cache_path=Path(args.scrape_cache) if args.scrape_cache else None,
#     )




if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--model", dest="model", default=DEFAULT_MODEL)
    ap.add_argument("--max-rows", dest="max_rows", type=int, default=None)
    ap.add_argument("--scrape-cache", dest="scrape_cache", default=None)
    args = ap.parse_args()

    run_stage2(
        Path(args.in_path),
        Path(args.out_path),
        model=args.model,
        max_rows=args.max_rows,
        scrape_cache_path=Path(args.scrape_cache) if args.scrape_cache else None,
    )
