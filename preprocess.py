import pandas as pd
from pathlib import Path

# --------------------------------------------------
# FedFit - WISDM Dataset Preprocessing
# --------------------------------------------------

DATA_PATH = Path("data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
OUTPUT_PATH = Path("data/wisdm_clean.csv")

# WISDM raw file has 6 columns
columns = [
    "user",
    "activity",
    "timestamp",
    "x_acc",
    "y_acc",
    "z_acc"
]

print("Loading WISDM dataset...")

# Read raw WISDM data
df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns,
    sep=",",
    engine="python",
    on_bad_lines="skip"
)

# Remove the semicolon from the final column
df["z_acc"] = (
    df["z_acc"]
    .astype(str)
    .str.replace(";", "", regex=False)
)

# Convert numeric columns
df["user"] = pd.to_numeric(df["user"], errors="coerce")
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df["x_acc"] = pd.to_numeric(df["x_acc"], errors="coerce")
df["y_acc"] = pd.to_numeric(df["y_acc"], errors="coerce")
df["z_acc"] = pd.to_numeric(df["z_acc"], errors="coerce")

# Remove invalid rows
df = df.dropna()

# Remove duplicate records
df = df.drop_duplicates()

# Sort by user and timestamp
df = df.sort_values(["user", "timestamp"])

# Reset index
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nPreprocessing completed successfully!")
print(f"Total records: {len(df):,}")
print(f"Users: {df['user'].nunique()}")
print(f"Activities: {df['activity'].nunique()}")

print("\nActivities:")
print(df["activity"].value_counts())

print("\nFirst 5 cleaned records:")
print(df.head())

print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")