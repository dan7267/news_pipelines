import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from joblib import load
from sentence_transformers import SentenceTransformer


# =========================
# Paths
# =========================

GOLD_BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "model_scored_daily"
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "interim" / "_state"
GDELT_DAILY_DIR = Path(__file__).resolve().parent.parent / "data" / "interim" / "gdelt_event_context_daily"

# Partner-friendly daily CSV drop (skip if already exists)
URLS_DIR = Path(__file__).resolve().parent.parent / "data" / "urls"

# Expert models root
EXPERT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "disruption_v2_experts"

# Score ONLY experts (skip general model)
EXPERT_TYPES = [
    "flood",
    "drought",
    "cyclone_hurricane",
    "extreme_heat",
    "landslide",
    "earthquake",
    "mine_accident",
    "labour_strike",
    "protests",
    "trade_embargo",
    "country_relations",
    "tariffs",
]

NATURAL_TYPES = [
    "flood",
    "drought",
    "cyclone_hurricane",
    "extreme_heat",
    "landslide",
    "earthquake",
]

SOCIAL_TYPES = [
    "labour_strike",
    "protests",
    "trade_embargo",
    "country_relations",
    "tariffs",
]

BAD_TEXT_PATTERNS = [
    "your privacy", "privacy choices", "cookie", "consent", "gdpr",
    "subscribe", "sign in", "login", "access denied",
    "captcha", "#value", "value!"
]


def looks_like_garbage(s: str) -> bool:
    if not isinstance(s, str):
        return True
    s = s.lower().strip()
    if len(s) < 15:
        return True
    return any(p in s for p in BAD_TEXT_PATTERNS)


def url_to_text(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        path = urlparse(url).path
        path = path.replace("/", " ")
        path = re.sub(r"[-_]+", " ", path)
        path = re.sub(r"\.(html|htm|php|aspx|jsp)$", "", path, flags=re.IGNORECASE)
        path = re.sub(r"\b\d+\b", " ", path)
        return re.sub(r"\s+", " ", path).strip().lower()
    except Exception:
        return ""


def build_text(row: pd.Series, use_url_fallback: bool = True) -> str:
    title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    desc = str(row.get("meta_description", "")) if pd.notna(row.get("meta_description")) else ""
    main = " ".join(f"{title}. {desc}".split())

    if use_url_fallback and looks_like_garbage(main):
        return url_to_text(str(row.get("url_normalized", "")))
    return main


def has_usable_metadata(row: pd.Series) -> bool:
    title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    desc = str(row.get("meta_description", "")) if pd.notna(row.get("meta_description")) else ""
    main = " ".join(f"{title}. {desc}".split())
    return not looks_like_garbage(main)


def _load_gdelt_latlon(target_date: str) -> pd.DataFrame:
    """Return a url_normalized → (actiongeo_lat, actiongeo_lon) lookup from the
    enriched GDELT file for target_date, or an empty DataFrame if not found."""
    year, month, day = target_date[:4], target_date[4:6], target_date[6:8]
    enriched = GDELT_DAILY_DIR / year / month / day / f"{target_date}_event_context_deduped_enriched.csv"
    if not enriched.exists():
        print(f"Warning: enriched GDELT file not found at {enriched} — lat/lon will be empty.")
        return pd.DataFrame(columns=["url_normalized", "actiongeo_lat", "actiongeo_lon"])
    geo = pd.read_csv(
        enriched,
        usecols=["url_normalized", "actiongeo_lat", "actiongeo_lon"],
        encoding="utf-8",
        engine="python",
    )
    return geo.drop_duplicates(subset="url_normalized", keep="first")


def _load_expert_bundle(expert_type: str) -> dict:
    """
    Expected layout from training code:
      models/disruption_v2_experts/expert_<type>/disruption_<type>.joblib
    """
    model_path = EXPERT_MODELS_DIR / f"expert_{expert_type}" / f"disruption_{expert_type}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Expert model not found: {model_path}")
    return load(model_path)


def main(target_date: str, top_k: int = 0, force: bool = False):
    URLS_DIR.mkdir(parents=True, exist_ok=True)

    simple_csv = URLS_DIR / f"{target_date}.csv"
    if simple_csv.exists() and not force:
        print(f"Skipping {target_date}: already done ({simple_csv})")
        return

    year, month, day = target_date[:4], target_date[4:6], target_date[6:8]
    out_dir = GOLD_BASE_DIR / year / month / day
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = STATE_DIR / f"url_title_meta_cache_{target_date}_fixed.csv"
    if not in_csv.exists():
        print(f"Skipping: Cleaned cache {in_csv.name} not found.")
        return

    print(f"\n--- Scoring EXPERT disruption types for {target_date} ---")
    df = pd.read_csv(in_csv, encoding="utf-8", engine="python")

    latlon = _load_gdelt_latlon(target_date)
    if not latlon.empty:
        df = df.merge(latlon, on="url_normalized", how="left")

    # Track whether there was usable metadata before any URL fallback
    df["has_usable_metadata"] = df.apply(has_usable_metadata, axis=1)
    df["kept_no_metadata"] = False

    bundles = {}
    embed_model_name = None
    use_url_fallback = True

    for t in EXPERT_TYPES:
        b = _load_expert_bundle(t)
        bundles[t] = b

        b_embed = b.get("embed_model", "all-MiniLM-L6-v2")
        if embed_model_name is None:
            embed_model_name = b_embed
        elif b_embed != embed_model_name:
            raise RuntimeError(
                f"Embed model mismatch: expected {embed_model_name}, but {t} has {b_embed}. "
                "All expert models must use the same embedding model."
            )

        use_url_fallback = use_url_fallback or bool(b.get("use_url_fallback", True))

    # Build text for embedding
    df["text"] = df.apply(lambda r: build_text(r, use_url_fallback=use_url_fallback), axis=1)

    embedder = SentenceTransformer(embed_model_name or "all-MiniLM-L6-v2")
    print(f"Embedding {len(df)} rows using {embed_model_name} ...")
    X = embedder.encode(
        df["text"].astype(str).tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Score experts
    p_cols = []
    for t in EXPERT_TYPES:
        clf = bundles[t]["classifier"]
        thr = float(bundles[t].get("threshold", 0.5))
        probs = clf.predict_proba(X)[:, 1]

        p_col = f"p_{t}"
        k_col = f"keep_{t}"
        df[p_col] = probs
        df[k_col] = probs >= thr
        p_cols.append(p_col)

    # Aggregate views
    natural_p_cols = [f"p_{t}" for t in NATURAL_TYPES]
    social_p_cols = [f"p_{t}" for t in SOCIAL_TYPES]
    natural_k_cols = [f"keep_{t}" for t in NATURAL_TYPES]
    social_k_cols = [f"keep_{t}" for t in SOCIAL_TYPES]

    df["p_natural_max"] = df[natural_p_cols].max(axis=1)
    df["p_social_max"] = df[social_p_cols].max(axis=1)

    df["keep_natural_any"] = df[natural_k_cols].any(axis=1)
    df["keep_social_any"] = df[social_k_cols].any(axis=1)

    df["top_natural_type"] = df[natural_p_cols].idxmax(axis=1).str.replace("^p_", "", regex=True)
    df["top_social_type"] = df[social_p_cols].idxmax(axis=1).str.replace("^p_", "", regex=True)

    df["p_any_expert"] = df[p_cols].max(axis=1)
    df["top_expert"] = df[p_cols].idxmax(axis=1).str.replace("^p_", "", regex=True)
    df["top_expert_p"] = df["p_any_expert"]

    # Decision rules
    # 1. No usable metadata -> keep
    df.loc[~df["has_usable_metadata"], "decision"] = "keep_no_metadata"

    # 2. Natural fires -> keep
    natural_mask = df["has_usable_metadata"] & df["keep_natural_any"]
    df.loc[natural_mask, "decision"] = "keep_natural"

    # 3. Social fires and natural does NOT -> remove
    social_only_mask = (
        df["has_usable_metadata"]
        & (~df["keep_natural_any"])
        & df["keep_social_any"]
    )
    df.loc[social_only_mask, "decision"] = "remove_social"

    # 4. Everything else with usable metadata is uncertain -> keep
    uncertain_mask = df["decision"].isna()
    df.loc[uncertain_mask, "decision"] = "keep_uncertain"

    # Convenience booleans
    df["kept_no_metadata"] = df["decision"] == "keep_no_metadata"
    df["kept_natural"] = df["decision"] == "keep_natural"
    df["removed_social"] = df["decision"] == "remove_social"
    df["kept_uncertain"] = df["decision"] == "keep_uncertain"
    df["keep_for_next_stage"] = df["decision"].isin(
        ["keep_no_metadata", "keep_natural", "keep_uncertain"]
    )

    # Save full scored file
    scored_path = out_dir / f"{target_date}_experts_scored.csv"
    df.to_csv(scored_path, index=False)

    kept = df[df["keep_for_next_stage"]].copy()
    kept = kept.sort_values(
        by=["kept_natural", "kept_uncertain", "kept_no_metadata", "p_natural_max", "p_any_expert"],
        ascending=[False, False, False, False, False],
    )

    if top_k > 0:
        kept = kept.head(top_k)

    urls_path_txt = out_dir / f"{target_date}_interesting_urls_experts.txt"
    urls_path_csv = out_dir / f"{target_date}_interesting_urls_experts_only.csv"

    kept["url_normalized"].to_csv(urls_path_txt, index=False, header=False)

    cols = [
        "url_normalized",
        "decision",
        "top_expert",
        "top_expert_p",
        "top_natural_type",
        "p_natural_max",
        "top_social_type",
        "p_social_max",
        "has_usable_metadata",
    ]
    for geo_col in ["actiongeo_lat", "actiongeo_lon"]:
        if geo_col in kept.columns:
            cols.append(geo_col)

    missing = [c for c in cols if c not in kept.columns]
    if missing:
        raise RuntimeError(f"Expected columns missing from kept dataframe: {missing}")

    kept[cols].to_csv(urls_path_csv, index=False)
    kept[cols].to_csv(simple_csv, index=False)

    # Stats
    total_scored = len(df)
    removed_social = int(df["removed_social"].sum())
    kept_uncertain = int(df["kept_uncertain"].sum())
    kept_natural = int(df["kept_natural"].sum())
    kept_no_metadata = int(df["kept_no_metadata"].sum())
    total_kept = int(df["keep_for_next_stage"].sum())

    print(f"Success! Folder created: {out_dir}")
    print(f"Total scored:                {total_scored}")
    print(f"Removed as social only:      {removed_social}")
    print(f"Kept as uncertain:           {kept_uncertain}")
    print(f"Kept as natural:             {kept_natural}")
    print(f"Kept due to no metadata:     {kept_no_metadata}")
    print(f"Total kept for next stage:   {total_kept}")
    print(f"Scored CSV:                  {scored_path.name}")
    print(f"URLs CSV:                    {urls_path_csv.name}")
    print(f"Partner CSV:                 {simple_csv}\n")


if __name__ == "__main__":
    target_date = input("Enter date to score (YYYYMMDD): ").strip()
    main(target_date, top_k=0, force=False)