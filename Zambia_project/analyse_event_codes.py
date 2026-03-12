import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ----------------------------------------------------
# Paths
# ----------------------------------------------------

DATA_PATH = Path("data/processed/pipeline_runs/20260218_20260304/eventcode_summary/matched_events.csv")

OUTPUT_DIR = Path("data/processed/pipeline_runs/20260218_20260304/eventcode_summary/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------
# Load data
# ----------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Total event rows:", len(df))
print("Unique articles:", df["sourceurl"].nunique())


# ----------------------------------------------------
# FACT 1: Code dispersion
# ----------------------------------------------------

unique_eventcodes = df["eventcode"].nunique()
unique_basecodes = df["eventbasecode"].nunique()
unique_rootcodes = df["eventrootcode"].nunique()

print("\n--- CODE DISPERSION ---")
print("Unique event codes:", unique_eventcodes)
print("Unique base codes:", unique_basecodes)
print("Unique root codes:", unique_rootcodes)

top_codes = df["eventcode"].value_counts(normalize=True).head(10)

print("\nTop event codes (% share):")
print(top_codes)


# ----------------------------------------------------
# ENTROPY OF EVENT CODES
# ----------------------------------------------------

p = df["eventcode"].value_counts(normalize=True)
entropy = -(p * np.log2(p)).sum()

print("\n--- EVENT CODE ENTROPY ---")
print("Entropy:", entropy)


# ----------------------------------------------------
# FACT 2: Political interaction classes
# ----------------------------------------------------

quadclass_dist = df["quadclass"].value_counts(normalize=True)

print("\n--- QUADCLASS DISTRIBUTION ---")
print(quadclass_dist)


# ----------------------------------------------------
# FACT 3: Articles mapped to multiple event codes
# ----------------------------------------------------

codes_per_article = df.groupby("sourceurl")["eventcode"].nunique()

print("\n--- EVENT CODES PER ARTICLE ---")
print("Mean codes per article:", codes_per_article.mean())
print("Max codes per article:", codes_per_article.max())

codes_per_article.describe()


# ----------------------------------------------------
# Root code distribution
# ----------------------------------------------------

root_dist = df["eventrootcode"].value_counts().sort_index()

print("\n--- ROOT CODE DISTRIBUTION ---")
print(root_dist)


# ----------------------------------------------------
# Save summary tables
# ----------------------------------------------------

df["eventcode"].value_counts().to_csv(OUTPUT_DIR / "eventcode_counts.csv")
df["eventrootcode"].value_counts().to_csv(OUTPUT_DIR / "eventrootcode_counts.csv")
quadclass_dist.to_csv(OUTPUT_DIR / "quadclass_distribution.csv")
codes_per_article.to_csv(OUTPUT_DIR / "codes_per_article.csv")


# ----------------------------------------------------
# PLOTS
# ----------------------------------------------------

plt.figure(figsize=(8,5))
root_dist.plot(kind="bar")
plt.title("Distribution of GDELT Event Root Codes")
plt.xlabel("Root Code")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "root_code_distribution.png")


plt.figure(figsize=(8,5))
df["quadclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution of GDELT QuadClass")
plt.xlabel("QuadClass")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "quadclass_distribution.png")


plt.figure(figsize=(8,5))
df["goldsteinscale"].dropna().hist(bins=30)
plt.title("Distribution of Goldstein Scale")
plt.xlabel("Goldstein score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "goldstein_distribution.png")


print("\nAnalysis complete. Outputs saved to:", OUTPUT_DIR)