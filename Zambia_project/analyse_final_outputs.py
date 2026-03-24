import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. LOAD DATA
# =========================

BASE_DIR = Path(__file__).resolve().parent
PATH = BASE_DIR / "combined_2016_2026.csv"
OUT_DIR = BASE_DIR / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)

START_DATE = pd.Timestamp("2016-03-16")
VALID_ESG = ["Environmental", "Social", "Governance"]

df = pd.read_csv(PATH)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# =========================
# 2. BASIC CLEANING
# =========================

# Parse date
df["sqldate"] = pd.to_datetime(df["sqldate"].astype(str), format="%Y%m%d", errors="coerce")

# Standardise string columns
str_cols = [
    "sourceurl",
    "impact_level1", "impact_level2", "impact_level3",
    "mining_company", "region", "mine_name", "mineral_type",
    "scrape_status", "impact_evidence", "title"
]

for c in str_cols:
    if c in df.columns:
        df[c] = df[c].replace({np.nan: None})
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

# Optional confidence filters
if "in_zambia_confidence" in df.columns:
    df = df[df["in_zambia_confidence"].fillna(0) > 0]

if "mining_related_confidence" in df.columns:
    df = df[df["mining_related_confidence"].fillna(0) > 0]

# Restrict to chosen analysis period
df = df[df["sqldate"].notna()].copy()
df = df[df["sqldate"] >= START_DATE].copy()

print("Rows after optional filters and date restriction:", len(df))


# =========================
# 3. HELPERS
# =========================

def split_pipepipe(val):
    if pd.isna(val) or val is None:
        return []
    parts = [x.strip() for x in str(val).split(" || ")]
    return [x for x in parts if x]

def clean_esg(val):
    if pd.isna(val) or val is None:
        return np.nan
    val = str(val).strip()
    return val if val in VALID_ESG else np.nan

def save_series_csv(series, path, index_name, value_name="count"):
    series.rename_axis(index_name).reset_index(name=value_name).to_csv(path, index=False)

def pad_lists(row, cols):
    max_len = max(len(row[c]) for c in cols)
    for c in cols:
        row[c] = row[c] + [np.nan] * (max_len - len(row[c]))
    return row


# =========================
# 4. EXPLODE MULTI-IMPACT ROWS
# =========================

impact_cols = ["impact_level1", "impact_level2", "impact_level3"]

for c in impact_cols:
    if c in df.columns:
        df[c] = df[c].apply(split_pipepipe)

df_imp = df.copy()
df_imp = df_imp.apply(lambda row: pad_lists(row, impact_cols), axis=1)
df_imp = df_imp.explode(impact_cols, ignore_index=True)

# Keep rows with any impact classification
df_imp = df_imp[df_imp["impact_level1"].notna()].copy()

# Clean ESG category labels
df_imp["impact_level1_clean"] = df_imp["impact_level1"].apply(clean_esg)

# Keep only rows with valid ESG category for ESG-specific analysis
df_esg = df_imp[df_imp["impact_level1_clean"].notna()].copy()

print("Article rows:", len(df))
print("Impact rows after explode:", len(df_imp))
print("Valid ESG impact rows:", len(df_esg))


# =========================
# 5. ARTICLE-LEVEL ESG TABLE
# =========================
# One article counts once per ESG category, even if mentioned multiple times

article_esg = (
    df_esg[["sourceurl", "sqldate", "impact_level1_clean"]]
    .drop_duplicates()
    .copy()
)

article_esg["month"] = article_esg["sqldate"].dt.to_period("M").dt.to_timestamp()
article_esg["year"] = article_esg["sqldate"].dt.year


# =========================
# 6. CORE HEADLINE STATS
# =========================

total_article_rows = len(df)
total_unique_articles = df["sourceurl"].nunique() if "sourceurl" in df.columns else len(df)
total_impact_rows = len(df_imp)
total_valid_esg_impacts = len(df_esg)

articles_with_any_impact = (
    df_imp["sourceurl"].nunique() if "sourceurl" in df_imp.columns else len(df_imp)
)
pct_articles_with_impact = (
    100 * articles_with_any_impact / total_unique_articles if total_unique_articles else np.nan
)

print("\n=== HEADLINE STATS ===")
print("Total article rows:", total_article_rows)
print("Total unique articles:", total_unique_articles)
print("Total impact rows:", total_impact_rows)
print("Total valid ESG impact rows:", total_valid_esg_impacts)
print("Unique articles with at least one impact:", articles_with_any_impact)
print(f"% of unique articles with impact: {pct_articles_with_impact:.1f}%")


# =========================
# 7. ESG DISTRIBUTION
# =========================
# Article-level for clean headline counts

impact1_counts = (
    article_esg["impact_level1_clean"]
    .value_counts()
    .reindex(VALID_ESG, fill_value=0)
)

impact1_pct = (impact1_counts / impact1_counts.sum() * 100).round(1)

impact1_summary = pd.DataFrame({
    "count": impact1_counts,
    "pct": impact1_pct
})

print("\n=== ARTICLE-LEVEL ESG CATEGORY COUNTS ===")
print(impact1_summary)

# Impact-level taxonomy summaries
impact2_counts = df_imp["impact_level2"].value_counts(dropna=True)
impact3_counts = df_imp["impact_level3"].value_counts(dropna=True)

print("\n=== TOP IMPACT LEVEL 2 ===")
print(impact2_counts.head(15))

print("\n=== TOP IMPACT LEVEL 3 ===")
print(impact3_counts.head(20))


# =========================
# 8. TIME TRENDS
# =========================

monthly_impacts = (
    article_esg.groupby("month")
    .size()
)

monthly_by_esg = (
    article_esg.groupby(["month", "impact_level1_clean"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=VALID_ESG, fill_value=0)
)

yearly_impacts = (
    article_esg.groupby("year")
    .size()
)

yearly_by_esg = (
    article_esg.groupby(["year", "impact_level1_clean"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=VALID_ESG, fill_value=0)
)

print("\n=== YEARLY ARTICLE-LEVEL ESG COUNTS ===")
print(yearly_impacts)

print("\n=== YEARLY BY ESG CATEGORY ===")
print(yearly_by_esg)

if len(yearly_impacts) >= 2 and yearly_impacts.iloc[0] > 0:
    growth_pct = 100 * (yearly_impacts.iloc[-1] - yearly_impacts.iloc[0]) / yearly_impacts.iloc[0]
    print(f"\nGrowth from first year to last year: {growth_pct:.1f}%")


# =========================
# 9. REGION-LEVEL STATS
# =========================

if "region" in df_esg.columns:
    region_article_esg = (
        df_esg.dropna(subset=["region"])[["sourceurl", "region", "impact_level1_clean"]]
        .drop_duplicates()
    )

    region_counts = (
        region_article_esg["region"]
        .value_counts(dropna=True)
        .head(15)
    )

    region_by_esg = (
        region_article_esg
        .groupby(["region", "impact_level1_clean"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=VALID_ESG, fill_value=0)
    )
    region_by_esg["total"] = region_by_esg.sum(axis=1)
    region_by_esg = region_by_esg.sort_values("total", ascending=False)

    print("\n=== TOP REGIONS ===")
    print(region_counts)

    print("\n=== REGION x ESG ===")
    print(region_by_esg.head(15))


# =========================
# 10. COMPANY-LEVEL ESG PROFILES
# =========================

if "mining_company" in df_esg.columns:
    company_article_esg = (
        df_esg.dropna(subset=["mining_company"])[["sourceurl", "mining_company", "impact_level1_clean"]]
        .drop_duplicates()
    )

    company_counts = (
        company_article_esg["mining_company"]
        .value_counts(dropna=True)
        .head(15)
    )

    company_by_esg = (
        company_article_esg
        .groupby(["mining_company", "impact_level1_clean"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=VALID_ESG, fill_value=0)
    )
    company_by_esg["total"] = company_by_esg.sum(axis=1)
    company_by_esg = company_by_esg.sort_values("total", ascending=False)

    print("\n=== TOP COMPANIES BY ARTICLE COUNT ===")
    print(company_counts)

    print("\n=== COMPANY x ESG ===")
    print(company_by_esg.head(15))


# =========================
# 11. METALS / MINERALS (FIXED SPLITTING)
# =========================

import re

def split_minerals(val):
    if pd.isna(val) or val is None:
        return []
    
    val = str(val).lower()
    
    # Replace common separators with comma
    val = re.sub(r"[&/]", ",", val)
    val = val.replace(" and ", ",")
    
    # Split on comma
    parts = [x.strip() for x in val.split(",")]
    
    # Remove empty
    parts = [x for x in parts if x]
    
    return parts

if "mineral_type" in df.columns:
    mineral_df = df[["sourceurl", "mineral_type"]].copy()
    
    # Split properly
    mineral_df["mineral_type"] = mineral_df["mineral_type"].apply(split_minerals)
    
    # Explode
    mineral_df = mineral_df.explode("mineral_type")
    
    # Clean text
    mineral_df["mineral_type"] = mineral_df["mineral_type"].str.strip()
    mineral_df = mineral_df[mineral_df["mineral_type"].notna()]
    mineral_df = mineral_df[mineral_df["mineral_type"] != ""]
    
    # Optional: standardise common names
    mineral_df["mineral_type"] = mineral_df["mineral_type"].replace({
        "cu": "copper",
        "co": "cobalt"
    })
    
    # Deduplicate so one article doesn't overcount same mineral
    mineral_df = mineral_df.drop_duplicates(subset=["sourceurl", "mineral_type"])
    
    mineral_counts = mineral_df["mineral_type"].value_counts()
    
    print("\n=== TOP MINERAL TYPES (SPLIT CORRECTLY) ===")
    print(mineral_counts.head(15))


# =========================
# 12. CONFIDENCE METRICS
# =========================

conf_cols = ["in_zambia_confidence", "mining_related_confidence", "impact_confidence"]
for c in conf_cols:
    if c in df.columns:
        print(f"\n=== {c.upper()} SUMMARY ===")
        print(df[c].describe())


# =========================
# 13. SCRAPE STATUS STATS
# =========================

if "scrape_status" in df.columns:
    print("\n=== SCRAPE STATUS COUNTS ===")
    print(df["scrape_status"].value_counts(dropna=False))


# =========================
# 14. SAVE SUMMARY TABLES
# =========================

impact1_summary.to_csv(OUT_DIR / "impact_level1_summary.csv")

save_series_csv(impact2_counts, OUT_DIR / "impact_level2_counts.csv", "impact_level2")
save_series_csv(impact3_counts, OUT_DIR / "impact_level3_counts.csv", "impact_level3")
save_series_csv(yearly_impacts, OUT_DIR / "yearly_article_esg_counts.csv", "year")
save_series_csv(monthly_impacts, OUT_DIR / "monthly_article_esg_counts.csv", "month")

yearly_by_esg.to_csv(OUT_DIR / "yearly_by_esg.csv")
monthly_by_esg.to_csv(OUT_DIR / "monthly_by_esg.csv")

if "region" in df_esg.columns:
    save_series_csv(region_counts, OUT_DIR / "region_counts.csv", "region")
    region_by_esg.to_csv(OUT_DIR / "region_by_esg.csv")

if "mining_company" in df_esg.columns:
    save_series_csv(company_counts, OUT_DIR / "company_counts.csv", "mining_company")
    company_by_esg.to_csv(OUT_DIR / "company_by_esg.csv")

if "mineral_type" in df.columns:
    save_series_csv(mineral_counts, OUT_DIR / "mineral_counts.csv", "mineral_type")


# =========================
# 15. CHARTS
# =========================

# ---- Chart 1: ESG category counts
plt.figure(figsize=(8, 5))
impact1_counts.plot(kind="bar")
plt.title("Articles Mentioning Each ESG Category")
plt.xlabel("ESG category")
plt.ylabel("Number of articles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "chart_impact_level1_counts.png", dpi=300)
plt.close()

# ---- Chart 2: Top 15 impact types (Level 3)
top15_l3 = impact3_counts.head(15).sort_values()
plt.figure(figsize=(10, 6))
top15_l3.plot(kind="barh")
plt.title("Top 15 Specific Impact Types")
plt.xlabel("Count")
plt.ylabel("Impact type")
plt.tight_layout()
plt.savefig(OUT_DIR / "chart_top15_impact_level3.png", dpi=300)
plt.close()

# ---- Chart 3: Monthly impacts over time starting 2016-03-16
plt.figure(figsize=(12, 5))
monthly_impacts.plot()
plt.title("Articles Mentioning ESG Impacts Over Time")
plt.xlabel("Date")
plt.ylabel("Number of articles")
plt.xlim(START_DATE, monthly_impacts.index.max())
plt.tight_layout()
plt.savefig(OUT_DIR / "chart_monthly_impacts_from_2016_03_16.png", dpi=300)
plt.close()

# ---- Chart 4: Monthly trends by ESG category starting 2016-03-16
plt.figure(figsize=(12, 6))
for col in VALID_ESG:
    plt.plot(monthly_by_esg.index, monthly_by_esg[col], label=col)
plt.title("ESG Article Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Number of articles")
plt.xlim(START_DATE, monthly_by_esg.index.max())
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "chart_monthly_by_esg_from_2016_03_16.png", dpi=300)
plt.close()

# ---- Chart 5: Top regions
if "region" in df_esg.columns and len(region_counts) > 0:
    top_regions = region_counts.head(10).sort_values()
    plt.figure(figsize=(9, 6))
    top_regions.plot(kind="barh")
    plt.title("Top 10 Regions by ESG Article Count")
    plt.xlabel("Number of articles")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_top_regions.png", dpi=300)
    plt.close()

# ---- Chart 6: Top companies
if "mining_company" in df_esg.columns and len(company_counts) > 0:
    top_companies = company_counts.head(10).sort_values()
    plt.figure(figsize=(9, 6))
    top_companies.plot(kind="barh")
    plt.title("Top 10 Mining Companies by ESG Article Count")
    plt.xlabel("Number of articles")
    plt.ylabel("Mining company")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_top_companies.png", dpi=300)
    plt.close()

# ---- Chart 7: Stacked company ESG profiles
if "mining_company" in df_esg.columns and len(company_by_esg) > 0:
    top_company_names = company_by_esg.head(8).index
    company_esg_top = (
        company_by_esg.loc[top_company_names, VALID_ESG]
    )

    company_esg_top.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("ESG Profiles of Top Mining Companies")
    plt.xlabel("Mining company")
    plt.ylabel("Number of articles")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_company_esg_profiles.png", dpi=300)
    plt.close()

# ---- Chart 8: Metals / minerals
if "mineral_type" in df.columns and len(mineral_counts) > 0:
    top_minerals = mineral_counts.head(15).sort_values()
    plt.figure(figsize=(10, 6))
    top_minerals.plot(kind="barh")
    plt.title("Most Frequently Mentioned Minerals / Metals")
    plt.xlabel("Number of articles")
    plt.ylabel("Mineral / metal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_metal_counts.png", dpi=300)
    plt.close()

print(f"\nDone. Outputs saved to: {OUT_DIR}")