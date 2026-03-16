from __future__ import annotations

import os
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


# ------------------ CONFIG ------------------ #

IN_PATH = Path(
    "data/interim/gdelt_event_context_daily/2025/12/01/zambia_events_collapsed_enriched.csv"
)

OUT_PATH = IN_PATH.with_name(
    IN_PATH.stem + "_mining_filtered.csv"
)

DEFAULT_MODEL = "gpt-5-mini"
LLM_MAX_WORKERS = 8


# ------------------ OPENAI SETUP ------------------ #

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found.")


# ------------------ HELPERS ------------------ #

def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def llm_definitely_not_mining(
    url: str,
    title: str,
    description: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 45,
) -> Dict[str, Any]:
    """
    Conservative removal logic:
    Only return True if article is clearly unrelated to mining.
    """

    title = _norm_str(title)
    description = _norm_str(description)

    # If no content, do NOT remove (conservative)
    if not title and not description:
        return {
            "definitely_not_mining": False,
            "confidence": 0.0,
        }

    system_prompt = (
        "You are a conservative classification engine. "
        "Return ONE JSON object and NOTHING else."
    )

    user_prompt = f"""
We are filtering news articles for a Zambia mining dataset.

Task:
Decide whether this article is DEFINITELY NOT related to mining.

Use ONLY the TITLE and DESCRIPTION below.

Definition of mining-related (broad):
- mining, extraction, exploration, drilling
- mine sites, pits, shafts, tailings
- mineral/metal commodities (copper, cobalt, lithium, nickel, gold, manganese, etc.)
- mining regulation, licensing, royalties, permits
- mine accidents, strikes, shutdowns
- transport/export of minerals or ore

IMPORTANT:
- Be VERY conservative.
- Only return true if it is clearly unrelated (e.g. football, celebrity gossip, unrelated crime, health, general politics with no mineral link).
- If uncertain, return false.

Output JSON with exactly:
{{
  "definitely_not_mining": true|false,
  "confidence": 0.0
}}

Confidence should only be >0.9 when it is unambiguously unrelated.

URL: {url}

TITLE: {title}
DESCRIPTION: {description}
""".strip()

    # Create client inside the function for safer threaded usage
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        timeout=timeout,
    )

    raw = completion.choices[0].message.content or "{}"
    data = json.loads(raw)

    try:
        flag = bool(data.get("definitely_not_mining", False))
    except Exception:
        flag = False

    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0

    conf = max(0.0, min(1.0, conf))

    return {
        "definitely_not_mining": flag,
        "confidence": conf,
    }


# ------------------ MAIN ------------------ #

def run_filter(
    in_path: Path = IN_PATH,
    out_path: Path = OUT_PATH,
    model: str = DEFAULT_MODEL,
    max_rows: Optional[int] = None,
) -> None:

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)

    if max_rows:
        df = df.head(max_rows).copy()

    if df.empty:
        print(f"No rows to classify in {in_path}")
        df["definitely not mining"] = pd.Series(dtype=bool)
        df["mining_confidence"] = pd.Series(dtype=float)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")
        return

    # Ensure output columns exist
    df["definitely not mining"] = False
    df["mining_confidence"] = 0.0

    rows = df.to_dict(orient="records")
    total_rows = len(rows)

    print(f"Running conservative mining filter on {total_rows:,} rows...")
    print(f"LLM workers: {LLM_MAX_WORKERS}")

    start_time = time.time()

    def classify_one(idx_row):
        idx, r = idx_row
        url = _norm_str(r.get("sourceurl"))
        title = _norm_str(r.get("title"))
        desc = _norm_str(r.get("description"))

        result = llm_definitely_not_mining(
            url=url,
            title=title,
            description=desc,
            model=model,
        )
        return idx, result

    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as ex:
        futures = [ex.submit(classify_one, (i, r)) for i, r in enumerate(rows)]

        completed = 0
        pbar = tqdm(total=total_rows, desc="definitely-not-mining filter")

        for fut in as_completed(futures):
            idx, result = fut.result()

            rows[idx]["definitely not mining"] = result["definitely_not_mining"]
            rows[idx]["mining_confidence"] = round(result["confidence"], 3)

            completed += 1
            pbar.update(1)

            if completed % 25 == 0 or completed == total_rows:
                elapsed = time.time() - start_time
                sec_per_row = elapsed / completed
                rows_remaining = total_rows - completed
                eta_seconds = rows_remaining * sec_per_row

                pbar.set_postfix({
                    "elapsed": format_eta(elapsed),
                    "eta": format_eta(eta_seconds),
                    "s/row": f"{sec_per_row:.2f}",
                })

        pbar.close()

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    total_elapsed = time.time() - start_time

    print(f"\nSaved: {out_path}")
    print("Rows flagged for removal:",
          int(out_df["definitely not mining"].sum()))
    print(f"First-classifier total time: {format_eta(total_elapsed)}")


def keep_remaining(
    in_path: Path,
    out_path: Path,
) -> Path:
    """Apply the notebook's remaining filter step.

    Keeps rows that are NOT flagged as definitely-not-mining OR rows where fetch_error is non-empty.
    """

    df = pd.read_csv(in_path)

    # Ensure fetch_error is treated as string
    df["fetch_error"] = df.get("fetch_error", "").fillna("").astype(str)

    filtered_df = df[
        (df["definitely not mining"] == False) |
        (df["fetch_error"].str.strip() != "")
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(out_path, index=False)

    print("Original rows:", len(df))
    print("Rows kept:", len(filtered_df))
    print("Rows removed:", len(df) - len(filtered_df))
    print("Saved to:", out_path)

    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--model", dest="model", default=DEFAULT_MODEL)
    ap.add_argument("--max-rows", dest="max_rows", type=int, default=None)
    ap.add_argument("--remaining-out", dest="remaining_out", default=None)
    args = ap.parse_args()

    run_filter(
        in_path=Path(args.in_path),
        out_path=Path(args.out_path),
        model=args.model,
        max_rows=args.max_rows,
    )
    if args.remaining_out:
        keep_remaining(Path(args.out_path), Path(args.remaining_out))