from pathlib import Path
import html
import pandas as pd
from ftfy import fix_text

# State files stay in the flat _state folder for easy cross-referencing
STATE_DIR = Path("data/interim/_state")

# --- Mojibake Mapping (Unchanged logic) ---
EM_DASH_BAD = "\u00e2\u20ac\u201d"
EN_DASH_BAD = "\u00e2\u20ac\u201c"

CONTROL_MAP = {
    "â€\x91": "‘", "â€\x92": "’", "â€\x93": "–", "â€\x94": "—",
    "â€\x85": "…", "â€\x9c": "“", "â€\x9d": "”",
}

LITERAL_MAP = {
    "â€“": "–", "â€”": "—", "â€˜": "‘", "â€™": "’", "â€œ": "“",
    "â€": "”", "â€¦": "…", "â„¢": "™", "Â£": "£", "Â€": "€",
    "Â ": " ", "\u00A0": " ",
}

def _try_redecode(s: str) -> str:
    suspicious = any(c in s for c in ("â", "Â", "Ã"))
    if not suspicious: return s
    for enc in ["cp1252", "latin-1"]:
        try:
            s2 = s.encode(enc, errors="strict").decode("utf-8", errors="strict")
            if sum(s2.count(c) for c in ("â", "Â", "Ã")) < sum(s.count(c) for c in ("â", "Â", "Ã")):
                return s2
        except: pass
    return s

def fix_meta_str(x):
    if pd.isna(x): return x
    s = str(x)
    s = html.unescape(s)
    s = fix_text(s)
    s = s.replace(EM_DASH_BAD, "—").replace(EN_DASH_BAD, "–")
    for bad, good in CONTROL_MAP.items(): s = s.replace(bad, good)
    for bad, good in LITERAL_MAP.items(): s = s.replace(bad, good)
    
    if any(c in s for c in ("â", "Â", "Ã")):
        s2 = _try_redecode(s)
        if s2 != s:
            s = fix_text(s2)
            # Re-apply maps after re-decoding
            for m in [CONTROL_MAP, LITERAL_MAP]:
                for bad, good in m.items(): s = s.replace(bad, good)
                
    return " ".join(s.split())

def main(target_date: str):
    """
    Cleans the title and meta description cache for a specific date.
    """
    in_file = STATE_DIR / f"url_title_meta_cache_{target_date}.csv"
    out_file = STATE_DIR / f"url_title_meta_cache_{target_date}_fixed.csv"

    if not in_file.exists():
        print(f"Skipping Step 4: {in_file.name} not found.")
        return

    print(f"Cleaning encoding issues for {target_date}...")
    
    # Use low_memory=False to handle mixed types in large caches
    df = pd.read_csv(
    in_file,
    engine="python",        # more tolerant parser
    on_bad_lines="skip"     # skip malformed rows
    )         

    for col in ["title", "meta_description"]:
        if col in df.columns:
            df[col] = df[col].apply(fix_meta_str)

    df.to_csv(out_file, index=False)

    # Validation
    t_bad = df["title"].astype(str).str.contains(r"[âÂÃ]", na=False).sum() if "title" in df.columns else 0
    print(f"Done! Cleaned cache saved to: {out_file.name}")
    print(f"Remaining suspicious markers: {t_bad}")

if __name__ == "__main__":
    day = input("Enter date to fix (YYYYMMDD): ").strip()
    main(day)