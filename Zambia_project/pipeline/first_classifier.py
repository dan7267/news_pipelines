# stage_mining_filter.py
# Conservative LLM filter: flag ONLY articles that are DEFINITELY NOT mining-related
# Adds two columns:
#   - "definitely not mining" (True/False)
#   - "mining_confidence" (0–1)

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# ------------------ CONFIG ------------------ #

IN_PATH = Path(
    "data/interim/gdelt_event_context_daily/2025/12/01/zambia_events_collapsed_enriched.csv"
)

OUT_PATH = IN_PATH.with_name(
    IN_PATH.stem + "_mining_filtered.csv"
)

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"


# ------------------ OPENAI SETUP ------------------ #

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found.")

client = OpenAI(api_key=api_key)


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

    # Ensure output columns exist
    df["definitely not mining"] = False
    df["mining_confidence"] = 0.0

    rows = df.to_dict(orient="records")

    print(f"Running conservative mining filter on {len(rows):,} rows...")

    for i, r in enumerate(tqdm(rows)):
        url = _norm_str(r.get("sourceurl"))
        title = _norm_str(r.get("title"))
        desc = _norm_str(r.get("description"))

        result = llm_definitely_not_mining(
            url=url,
            title=title,
            description=desc,
            model=model,
        )

        r["definitely not mining"] = result["definitely_not_mining"]
        r["mining_confidence"] = round(result["confidence"], 3)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSaved: {out_path}")
    print("Rows flagged for removal:",
          int(out_df["definitely not mining"].sum()))



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

    filtered_df = df[(df["definitely not mining"] == False) | (df["fetch_error"].str.strip() != "")]

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

    out = run_filter(in_path=Path(args.in_path), out_path=Path(args.out_path), model=args.model, max_rows=args.max_rows)
    if args.remaining_out:
        keep_remaining(Path(args.out_path), Path(args.remaining_out))


## If title, description are empty or missing, it outputs FALSE and confidence is 0.0s