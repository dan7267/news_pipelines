import re
import io
import csv
import zipfile
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

MASTER = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
TARGET_SUFFIX = ".export.CSV.zip"

# One file per day
OUT_DIR = Path("data/interim/gdelt_event_context_daily")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Track processed 15-min files so reruns don't duplicate
STATE_DIR = Path("data/interim/_state/gdelt")
STATE_DIR.mkdir(parents=True, exist_ok=True)

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
    """
    GDELT exports should match the schema length; this makes the pipeline robust:
    - if shorter, pad with ""
    - if longer, trim extras
    """
    if len(row) < n:
        return row + [""] * (n - len(row))
    if len(row) > n:
        return row[:n]
    return row


def extract_rows_from_zip(url: str) -> List[List[str]]:
    ingest_dt = url_timestamp(url)
    ingest_time = ingest_dt.strftime("%Y%m%d%H%M%S") if ingest_dt else ""

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    buf = io.BytesIO(r.content)

    out_rows: List[List[str]] = []
    with zipfile.ZipFile(buf) as zf:
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
                out_rows.append([ingest_time] + row)

    return out_rows


def main(target_day: str) -> None:
    """
    target_day: 'YYYYMMDD' (e.g., '20260224')
    """
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
        print(f"No files found for date: {target_day}")
        return

    targets.sort()
    print(f"Found {len(targets)} files for {target_day}. Processing...")

    for ts, url in targets:
        marker = processed_marker(ts)
        if marker.exists():
            print(f"Skipping (already done): {url.split('/')[-1]}")
            continue

        out_path = daily_output_path(ts)
        ensure_header(out_path)

        print(f"Processing: {ts.strftime('%H:%M')} -> {url.split('/')[-1]}")
        extracted = extract_rows_from_zip(url)

        with open(out_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(extracted)

        marker.touch()

    print(f"Done! Daily file is at: {daily_output_path(targets[0][0])}")


if __name__ == "__main__":
    day_to_process = input("Enter date to process (YYYYMMDD): ").strip()
    main(day_to_process)