from pathlib import Path
import pandas as pd

# Get the folder where this script lives
BASE_DIR = Path(__file__).resolve().parent

# Point to chunk_outputs inside it
ROOT = BASE_DIR / "chunk_outputs"
OUTPUT = BASE_DIR / "combined_2016_2026.csv"

csv_paths = sorted(ROOT.glob("*/*/final.csv"))

if not csv_paths:
    raise FileNotFoundError(f"No CSV files found under {ROOT}")

first = True
total_rows = 0
files_done = 0

for path in csv_paths:
    try:
        df = pd.read_csv(path)

        # Optional but VERY useful for debugging later
        df["year_folder"] = path.parent.parent.name
        df["chunk_folder"] = path.parent.name

        df.to_csv(
            OUTPUT,
            mode="w" if first else "a",
            index=False,
            header=first,
        )

        first = False
        total_rows += len(df)
        files_done += 1

        print(f"Appended {path} ({len(df):,} rows)")

    except Exception as e:
        print(f"Failed on {path}: {e}")

print("\nDone.")
print(f"Files combined: {files_done}")
print(f"Total rows: {total_rows:,}")
print(f"Saved to: {OUTPUT}")