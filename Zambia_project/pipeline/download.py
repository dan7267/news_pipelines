import csv
import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, List, Optional, Tuple

import requests

MASTER = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
TARGET_SUFFIX = ".export.CSV.zip"

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent

OUT_DIR = PROJECT_ROOT / "data" / "interim" / "gdelt_event_context_daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_DIR = PROJECT_ROOT / "data" / "interim" / "_state" / "gdelt"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Conservative worker count to avoid memory / bandwidth spikes.
MAX_WORKERS = 4
CHUNK_SIZE = 1024 * 1024
TIMEOUT = (15, 120)

# --- Full GDELT Events export schema (standard order) ---
# We add ingest_time as an extra column at the front.
EVENT_FIELDS = [
    "globaleventid",
    "sqldate",
    "monthyear",
    "year",
    "fractiondate",
    "actor1code",
    "actor1name",
    "actor1countrycode",
    "actor1knowngroupcode",
    "actor1ethniccode",
    "actor1religion1code",
    "actor1religion2code",
    "actor1type1code",
    "actor1type2code",
    "actor1type3code",
    "actor2code",
    "actor2name",
    "actor2countrycode",
    "actor2knowngroupcode",
    "actor2ethniccode",
    "actor2religion1code",
    "actor2religion2code",
    "actor2type1code",
    "actor2type2code",
    "actor2type3code",
    "isrootevent",
    "eventcode",
    "eventbasecode",
    "eventrootcode",
    "quadclass",
    "goldsteinscale",
    "numentions",
    "numsources",
    "numarticles",
    "avgtone",
    "actor1geo_type",
    "actor1geo_fullname",
    "actor1geo_countrycode",
    "actor1geo_adm1code",
    "actor1geo_adm2code",
    "actor1geo_lat",
    "actor1geo_lon",
    "actor1geo_featureid",
    "actor2geo_type",
    "actor2geo_fullname",
    "actor2geo_countrycode",
    "actor2geo_adm1code",
    "actor2geo_adm2code",
    "actor2geo_lat",
    "actor2geo_lon",
    "actor2geo_featureid",
    "actiongeo_type",
    "actiongeo_fullname",
    "actiongeo_countrycode",
    "actiongeo_adm1code",
    "actiongeo_adm2code",
    "actiongeo_lat",
    "actiongeo_lon",
    "actiongeo_featureid",
    "dateadded",
    "sourceurl",
]

HEADER = ["ingest_time"] + EVENT_FIELDS

ZAMBIA_GEO_FIPS = "ZA"
ZAMBIA_ACTOR_CAMEO = "ZMB"

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

EVENT_INDEX = {name: i for i, name in enumerate(EVENT_FIELDS)}


def _field(row: List[str], name: str) -> str:
    idx = EVENT_INDEX[name]
    if idx >= len(row):
        return ""
    return (row[idx] or "").strip()


def row_is_zambia(row: List[str]) -> bool:
    geo_country_cols = ["actiongeo_countrycode", "actor1geo_countrycode", "actor2geo_countrycode"]
    geo_adm1_cols = ["actiongeo_adm1code", "actor1geo_adm1code", "actor2geo_adm1code"]
    geo_full_cols = ["actiongeo_fullname", "actor1geo_fullname", "actor2geo_fullname"]
    actor_country_cols = ["actor1countrycode", "actor2countrycode"]
    actor_name_cols = ["actor1name", "actor2name"]

    for c in geo_country_cols:
        if _field(row, c) == ZAMBIA_GEO_FIPS:
            return True

    for c in geo_adm1_cols:
        if _field(row, c).startswith(ZAMBIA_GEO_FIPS):
            return True

    for c in actor_country_cols:
        if _field(row, c) == ZAMBIA_ACTOR_CAMEO:
            return True

    for c in geo_full_cols:
        text = _field(row, c).lower()
        if any(term in text for term in TEXT_TERMS):
            return True

    for c in actor_name_cols:
        text = _field(row, c).lower()
        if "zambia" in text or "zambian" in text:
            return True

    return False


def parse_masterfile(text: str) -> List[Tuple[int, str, str]]:
    rows: List[Tuple[int, str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        size, md5, url = parts[0], parts[1], parts[2]
        try:
            rows.append((int(size), md5, url))
        except ValueError:
            continue
    return rows


def url_timestamp(url: str) -> Optional[datetime]:
    m = re.search(r"/(\d{14})\.", url)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")


def processed_marker(ts: datetime) -> Path:
    return STATE_DIR / f"{ts.strftime('%Y%m%d%H%M%S')}.done"


def daily_output_path(ts: datetime) -> Path:
    year = ts.strftime("%Y")
    month = ts.strftime("%m")
    day_folder = ts.strftime("%d")
    day_file = ts.strftime("%Y%m%d")

    out_dir = OUT_DIR / year / month / day_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{day_file}_events_full.csv"


def ensure_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HEADER)


def pad_or_trim(row: List[str], n: int) -> List[str]:
    if len(row) < n:
        return row + [""] * (n - len(row))
    if len(row) > n:
        return row[:n]
    return row


def _stream_zip_to_tempfile(url: str) -> Path:
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_path = Path(tmp.name)
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    tmp.write(chunk)
    return tmp_path


def iter_rows_from_local_zip(zip_path: Path, url: str) -> Iterator[List[str]]:
    ingest_dt = url_timestamp(url)
    ingest_time = ingest_dt.strftime("%Y%m%d%H%M%S") if ingest_dt else ""

    with zipfile.ZipFile(zip_path) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as f_in:
            reader = csv.reader(
                io.TextIOWrapper(f_in, encoding="utf-8", errors="replace"),
                delimiter="\t",
            )
            for row in reader:
                if not row:
                    continue
                row = pad_or_trim(row, len(EVENT_FIELDS))
                if row_is_zambia(row):
                    yield [ingest_time] + row


def download_targets(targets: List[Tuple[datetime, str]]) -> dict[datetime, Path]:
    download_jobs = []
    for ts, url in targets:
        marker = processed_marker(ts)
        if marker.exists():
            continue
        download_jobs.append((ts, url))

    downloaded: dict[datetime, Path] = {}
    if not download_jobs:
        return downloaded

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {
            ex.submit(_stream_zip_to_tempfile, url): (ts, url)
            for ts, url in download_jobs
        }
        for fut in as_completed(future_map):
            ts, url = future_map[fut]
            try:
                downloaded[ts] = fut.result()
                print(f"Downloaded: {url.split('/')[-1]}", flush=True)
            except Exception as e:
                print(f"Failed download: {url} | {e}", flush=True)
                raise

    return downloaded


def main(target_day: str) -> None:
    print(f"Target day: {target_day}", flush=True)
    print("Downloading GDELT masterfile...", flush=True)
    master_txt = requests.get(MASTER, timeout=60).text

    rows = parse_masterfile(master_txt)
    targets: List[Tuple[datetime, str]] = []

    for _, _, url in rows:
        if not url.endswith(TARGET_SUFFIX):
            continue
        ts = url_timestamp(url)
        if ts and ts.strftime("%Y%m%d") == target_day:
            targets.append((ts, url))

    if not targets:
        print(f"No files found for date: {target_day}", flush=True)
        return

    targets.sort()
    downloaded = download_targets(targets)

    for ts, url in targets:
        marker = processed_marker(ts)
        if marker.exists():
            continue

        out_path = daily_output_path(ts)
        ensure_header(out_path)

        zip_path = downloaded.get(ts)
        if zip_path is None:
            # Fallback in case a target was not predownloaded for some reason.
            zip_path = _stream_zip_to_tempfile(url)

        try:
            with open(out_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for out_row in iter_rows_from_local_zip(zip_path, url):
                    writer.writerow(out_row)
            marker.touch()
        finally:
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Date to process in YYYYMMDD format")
    args = ap.parse_args()

    main(args.date)
