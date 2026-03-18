import re
import io
import csv
import zipfile
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict


MASTER = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
TARGET_SUFFIX = ".export.CSV.zip"

# One file per day
OUT_DIR = Path("data/interim/gdelt_event_context_daily")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Track processed 15-min files so reruns don't duplicate
STATE_DIR = Path("data/interim/_state/gdelt")
STATE_DIR.mkdir(parents=True, exist_ok=True)


HEADER = [
    "globaleventid",
    "sqldate",
    "ingest_time",
    "eventcode",
    "eventbasecode",
    "eventrootcode",
    "quadclass",
    "numentions",
    "numsources",
    "numarticles",
    "avgtone",
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


def parse_masterfile(text: str) -> List[Tuple[int, str, str]]:
    rows = []
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


def safe_get(row: List[str], idx: int, default: str = "") -> str:
    # supports negative indices too
    try:
        return row[idx]
    except Exception:
        return default


def processed_marker(ts: datetime) -> Path:
    # one marker file per 15-min interval
    return STATE_DIR / f"{ts.strftime('%Y%m%d%H%M%S')}.done"


def daily_output_path(ts: datetime) -> Path:
    """
    Creates and returns path for data/interim/gdelt_event_context_daily/YYYY/MM/DD/
    """
    year = ts.strftime("%Y")
    month = ts.strftime("%m")
    day_folder = ts.strftime("%d")  # Extract just the day for the folder name
    day_file = ts.strftime("%Y%m%d") # For the filename

    # Build the full directory path
    out_dir = OUT_DIR / year / month / day_folder
    
    # Create the nested directory structure if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    
    return out_dir / f"{day_file}_event_context.csv"


def ensure_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def extract_rows_from_zip(url: str) -> List[List[str]]:
    """
    Returns extracted rows (as list-of-fields) from one zipped export file.
    We return structured fields already mapped to HEADER order.
    """
    ingest_dt = url_timestamp(url)
    ingest_time = ingest_dt.strftime("%Y%m%d%H%M%S") if ingest_dt else ""

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    buf = io.BytesIO()
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            buf.write(chunk)
    buf.seek(0)

    out_rows = []

    with zipfile.ZipFile(buf) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as f_in:
            reader = csv.reader(io.TextIOWrapper(f_in, encoding="utf-8", errors="replace"), delimiter="\t")

            for row in reader:
                if not row or len(row) < 10:
                    continue

                sourceurl = safe_get(row, -1)
                if not sourceurl:
                    continue

                # Tail fields (robust)
                actiongeo_fullname = safe_get(row, -9)
                actiongeo_country = safe_get(row, -8)
                actiongeo_adm1 = safe_get(row, -7)
                actiongeo_adm2 = safe_get(row, -6)
                actiongeo_lat = safe_get(row, -5)
                actiongeo_lon = safe_get(row, -4)
                actiongeo_featureid = safe_get(row, -3)
                dateadded = safe_get(row, -2)

                # Core fields (common indices for Events export)
                globaleventid = safe_get(row, 0)
                sqldate = safe_get(row, 1)
                eventcode = safe_get(row, 26)
                eventbasecode = safe_get(row, 27)
                eventrootcode = safe_get(row, 28)
                quadclass = safe_get(row, 29)
                numentions = safe_get(row, 31)
                numsources = safe_get(row, 32)
                numarticles = safe_get(row, 33)
                avgtone = safe_get(row, 34)

                out_rows.append([
                    globaleventid,
                    sqldate,
                    ingest_time,
                    eventcode,
                    eventbasecode,
                    eventrootcode,
                    quadclass,
                    numentions,
                    numsources,
                    numarticles,
                    avgtone,
                    actiongeo_fullname,
                    actiongeo_country,
                    actiongeo_adm1,
                    actiongeo_adm2,
                    actiongeo_lat,
                    actiongeo_lon,
                    actiongeo_featureid,
                    dateadded,
                    sourceurl,
                ])

    return out_rows


def main(target_day: str) -> None:
    """
    target_day: format 'YYYYMMDD' (e.g., '20230501')
    """
    # 1. Fetch the master list
    master_txt = requests.get(MASTER, timeout=60).text
    rows = parse_masterfile(master_txt)

    # 2. Filter for files matching that specific day
    targets = []
    for _, _, url in rows:
        if not url.endswith(TARGET_SUFFIX):
            continue
        
        # Extract the timestamp from the URL
        ts = url_timestamp(url)
        if ts and ts.strftime("%Y%m%d") == target_day:
            targets.append((ts, url))

    if not targets:
        print(f"No files found for date: {target_day}")
        return

    targets.sort()
    print(f"Found {len(targets)} files for {target_day}. Processing...")

    # 3. Process and concatenate
    for ts, url in targets:
        marker = processed_marker(ts)
        if marker.exists():
            print(f"Skipping (already done): {url.split('/')[-1]}")
            continue

        out_path = daily_output_path(ts)
        ensure_header(out_path)

        print(f"Processing: {ts.strftime('%H:%M')}")
        extracted = extract_rows_from_zip(url)

        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(extracted)

        marker.touch()
    
    print(f"Done! Daily file is at: {daily_output_path(targets[0][0])}")

if __name__ == "__main__":
    day_to_process = input("Enter date to process (YYYYMMDD): ").strip()
    main(day_to_process)