import csv
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import requests


MASTER = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
TARGET_SUFFIX = ".export.CSV.zip"
CHUNK_SIZE = 1024 * 1024  # 1 MB
REQUEST_TIMEOUT = (20, 120)


OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "interim" / "gdelt_event_context_daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Track processed 15-min files so reruns don't duplicate
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "interim" / "_state" / "gdelt"
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


def safe_get(row: List[str], idx: int, default: str = "") -> str:
    try:
        return row[idx]
    except Exception:
        return default


def processed_marker(ts: datetime) -> Path:
    return STATE_DIR / f"{ts.strftime('%Y%m%d%H%M%S')}.done"


def daily_output_path(ts: datetime) -> Path:
    year = ts.strftime("%Y")
    month = ts.strftime("%m")
    day_folder = ts.strftime("%d")
    day_file = ts.strftime("%Y%m%d")

    out_dir = OUT_DIR / year / month / day_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{day_file}_event_context.csv"


def ensure_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def iter_extracted_rows_from_zipfile(zip_path: Path, ingest_time: str) -> Iterator[List[str]]:
    """
    Stream rows from a downloaded GDELT zip on disk.
    This avoids holding the whole decompressed interval in memory.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not names:
            return

        inner = names[0]
        with zf.open(inner) as f_in:
            text_stream = (line.decode("utf-8", errors="replace") for line in f_in)
            reader = csv.reader(text_stream, delimiter="\t")

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

                yield [
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
                ]


def append_zip_rows_to_daily_csv(url: str, out_path: Path, session: requests.Session) -> int:
    """
    Download one zip to a temporary file on disk, then stream rows from it
    directly into the daily CSV. Returns number of rows written.
    """
    ingest_dt = url_timestamp(url)
    ingest_time = ingest_dt.strftime("%Y%m%d%H%M%S") if ingest_dt else ""

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    tmp.write(chunk)
        tmp.flush()

        rows_written = 0
        with open(out_path, "a", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            for extracted_row in iter_extracted_rows_from_zipfile(Path(tmp.name), ingest_time):
                writer.writerow(extracted_row)
                rows_written += 1

    return rows_written


def main(target_day: str) -> None:
    """
    target_day: format 'YYYYMMDD' (e.g., '20230501')
    """
    with requests.Session() as session:
        master_resp = session.get(MASTER, timeout=REQUEST_TIMEOUT)
        master_resp.raise_for_status()
        rows = parse_masterfile(master_resp.text)

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

        total_rows_written = 0
        for ts, url in targets:
            marker = processed_marker(ts)
            if marker.exists():
                print(f"Skipping (already done): {url.split('/')[-1]}")
                continue

            out_path = daily_output_path(ts)
            ensure_header(out_path)

            print(f"Processing: {ts.strftime('%H:%M')}")
            rows_written = append_zip_rows_to_daily_csv(url, out_path, session)
            print(f"  wrote {rows_written:,} rows")
            total_rows_written += rows_written

            marker.touch()

    print(f"Done! Daily file is at: {daily_output_path(targets[0][0])}")
    print(f"Total rows written for {target_day}: {total_rows_written:,}")


if __name__ == "__main__":
    day_to_process = input("Enter date to process (YYYYMMDD): ").strip()
    main(day_to_process)