from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from joblib import dump

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================
# Config
# ============================
BASE_DIR = Path(__file__).resolve().parent
TRAINING_XLSX = BASE_DIR / "data" / "interim" / "social_natural_training.xlsx"
SHEET_NAME = "data"
OUTPUT_DIR = BASE_DIR / "models" / "social_vs_natural_v1"

USE_URL_FALLBACK = True
REQUIRE_TEXT = True
THRESHOLD = 0.5  # probability threshold for predicting "social"

ROW_ORIGIN_COL = "row_origin"
GOLD_ORIGIN_VALUE = "gold_manual"

URL_COL = "url_normalized"
TITLE_COL = "title"
META_COL = "meta_description"

SOCIAL_COL = "social_disruption"
NATURAL_COL = "natural_disruption"

WEAK_SOCIAL_COL = "chatgpt_social_disruption"
WEAK_NATURAL_COL = "chatgpt_natural_disruption"

BAD_TEXT_PATTERNS = [
    "your privacy", "privacy choices", "cookie", "consent", "gdpr",
    "subscribe", "sign in", "login", "access denied",
    "captcha", "#value", "value!", "msn", "bot"
]


# ============================
# Utilities
# ============================
def to_int01(x) -> int:
    if pd.isna(x):
        return 0
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and x in (0, 1):
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return 1
    if s in {"0", "false", "no", "n", "f"}:
        return 0
    return 0


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
    except Exception:
        return ""
    path = path.replace("/", " ")
    path = re.sub(r"[-_]+", " ", path)
    path = re.sub(r"\.(html|htm|php|aspx|jsp)$", "", path, flags=re.IGNORECASE)
    path = re.sub(r"\b\d+\b", " ", path)
    path = re.sub(r"\s+", " ", path).strip().lower()
    return path


def build_text(row: pd.Series) -> str:
    title = "" if pd.isna(row.get(TITLE_COL)) else str(row.get(TITLE_COL))
    desc = "" if pd.isna(row.get(META_COL)) else str(row.get(META_COL))
    url = "" if pd.isna(row.get(URL_COL)) else str(row.get(URL_COL))

    main = f"{title}. {desc}".strip()
    main = " ".join(main.split())

    if USE_URL_FALLBACK and looks_like_garbage(main):
        return url_to_text(url)
    return main


def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")


# ============================
# Label construction
# ============================
def build_binary_social_vs_natural(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a clean binary dataset:
      label = 1 -> social disruption
      label = 0 -> natural disruption

    Gold rows use manual labels.
    Non-gold rows use weak labels if available.

    Rows are kept only if exactly one of social/natural is 1.
    """

    is_gold = (df[ROW_ORIGIN_COL].fillna("") == GOLD_ORIGIN_VALUE).values

    gold_social = df[SOCIAL_COL].map(to_int01).values
    gold_natural = df[NATURAL_COL].map(to_int01).values

    if WEAK_SOCIAL_COL in df.columns and WEAK_NATURAL_COL in df.columns:
        weak_social = df[WEAK_SOCIAL_COL].map(to_int01).values
        weak_natural = df[WEAK_NATURAL_COL].map(to_int01).values
    else:
        weak_social = gold_social
        weak_natural = gold_natural

    social = np.where(is_gold, gold_social, weak_social).astype(int)
    natural = np.where(is_gold, gold_natural, weak_natural).astype(int)

    out = df.copy()
    out["social_final"] = social
    out["natural_final"] = natural

    # Keep only rows with exactly one class active
    out = out[(out["social_final"] + out["natural_final"]) == 1].copy()

    # Binary target: social=1, natural=0
    out["label"] = out["social_final"].astype(int)

    return out


# ============================
# Main
# ============================
print("Loading Excel:", TRAINING_XLSX)
df = pd.read_excel(TRAINING_XLSX, sheet_name=SHEET_NAME, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
print("Rows loaded:", len(df))

required_cols = [
    ROW_ORIGIN_COL, URL_COL, TITLE_COL, META_COL,
    SOCIAL_COL, NATURAL_COL
]
ensure_columns(df, required_cols)

# Build text
df["text"] = df.apply(build_text, axis=1)
if REQUIRE_TEXT:
    before = len(df)
    df = df[df["text"].astype(str).str.len() > 0].copy().reset_index(drop=True)
    print(f"Dropped empty-text rows: {before} -> {len(df)}")
else:
    df = df.reset_index(drop=True)

# Build binary dataset
df_bin = build_binary_social_vs_natural(df).reset_index(drop=True)
print("Rows after keeping only clean social-vs-natural cases:", len(df_bin))
print("Label counts (1=social, 0=natural):", df_bin["label"].value_counts().to_dict())

if df_bin["label"].nunique() < 2:
    raise ValueError("Need both classes present to train.")

# Train/test split
idx = np.arange(len(df_bin))
idx_train, idx_test = train_test_split(
    idx,
    test_size=0.2,
    stratify=df_bin["label"].values,
    random_state=42,
)

X_train = df_bin.loc[idx_train, "text"].astype(str).tolist()
X_test = df_bin.loc[idx_test, "text"].astype(str).tolist()
y_train = df_bin.loc[idx_train, "label"].values
y_test = df_bin.loc[idx_test, "label"].values

# Embeddings
print("Embedding...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_train_emb = embedder.encode(X_train, normalize_embeddings=True, show_progress_bar=True)
X_test_emb = embedder.encode(X_test, normalize_embeddings=True, show_progress_bar=True)

# Classifier
clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5)
clf.fit(X_train_emb, y_train)

probs = clf.predict_proba(X_test_emb)[:, 1]   # p(social)
preds = (probs > THRESHOLD).astype(int)

print("\nAverage precision:", average_precision_score(y_test, probs))
print("\nPrecision:", precision_score(y_test, preds, zero_division=0))
print("Recall:", recall_score(y_test, preds, zero_division=0))
print("F1:", f1_score(y_test, preds, zero_division=0))
print(f"\nClassification report @ threshold={THRESHOLD}:")
print(classification_report(y_test, preds))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, preds))

# Error review
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

review_df = df_bin.loc[idx_test].copy()
review_df["p_social"] = probs
review_df["y_true"] = y_test
review_df["y_pred"] = preds
review_df["error_type"] = ""
review_df["true_class"] = np.where(review_df["y_true"] == 1, "social", "natural")
review_df["pred_class"] = np.where(review_df["y_pred"] == 1, "social", "natural")

review_df.loc[(review_df["y_true"] == 0) & (review_df["y_pred"] == 1), "error_type"] = "FALSE_SOCIAL"
review_df.loc[(review_df["y_true"] == 1) & (review_df["y_pred"] == 0), "error_type"] = "FALSE_NATURAL"

false_social = review_df[review_df["error_type"] == "FALSE_SOCIAL"].sort_values("p_social", ascending=False)
false_natural = review_df[review_df["error_type"] == "FALSE_NATURAL"].sort_values("p_social", ascending=True)

review_path = out_dir / "error_review.xlsx"
with pd.ExcelWriter(review_path, engine="openpyxl") as writer:
    review_df.sort_values("p_social", ascending=False).to_excel(writer, index=False, sheet_name="all_test")
    false_social.to_excel(writer, index=False, sheet_name="false_social")
    false_natural.to_excel(writer, index=False, sheet_name="false_natural")

print(f"\nWrote error review Excel to: {review_path}")

# Save model bundle
dump(
    {
        "embed_model": "all-MiniLM-L6-v2",
        "classifier": clf,
        "threshold": THRESHOLD,
        "label_meaning": {"1": "social", "0": "natural"},
        "use_url_fallback": USE_URL_FALLBACK,
    },
    out_dir / "social_vs_natural_model.joblib",
)

print("\nModel saved to:", out_dir / "social_vs_natural_model.joblib")