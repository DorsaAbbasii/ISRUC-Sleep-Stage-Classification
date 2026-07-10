from pathlib import Path
import numpy as np
import pandas as pd


INPUT_FILE = Path("isruc_features_clean_6ch.csv")
OUTPUT_FILE = Path("isruc_features_enhanced_6ch.csv")

CHANNELS = ["F3", "C3", "O1", "F4", "C4", "O2"]
LEFT_RIGHT_PAIRS = [("F3", "F4"), ("C3", "C4"), ("O1", "O2")]
BANDS = ["delta_power", "theta_power", "alpha_power", "beta_power"]

EPS = 1e-8

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Input file not found: isruc_features_clean_6ch.csv"
    )

print("Loading clean 6-channel features...")
df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# Make sure epochs are in correct order for temporal features
df = df.sort_values(["subject_group", "epoch"]).reset_index(drop=True)

# 1. Relative band power and ratios
for ch in CHANNELS:
    band_cols = [f"{ch}_{band}" for band in BANDS]

    missing = [col for col in band_cols if col not in df.columns]
    if missing:
        print(f"Skipping {ch}, missing columns:", missing)
        continue

    total_power = sum(df[col] for col in band_cols) + EPS

    for band in BANDS:
        df[f"{ch}_rel_{band}"] = df[f"{ch}_{band}"] / total_power

    df[f"{ch}_delta_theta_ratio"] = df[f"{ch}_delta_power"] / (df[f"{ch}_theta_power"] + EPS)
    df[f"{ch}_alpha_theta_ratio"] = df[f"{ch}_alpha_power"] / (df[f"{ch}_theta_power"] + EPS)
    df[f"{ch}_beta_alpha_ratio"] = df[f"{ch}_beta_power"] / (df[f"{ch}_alpha_power"] + EPS)

# 2. Left-right channel differences
base_feature_types = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "mean",
    "std",
    "energy",
]

for left, right in LEFT_RIGHT_PAIRS:
    for feat in base_feature_types:
        left_col = f"{left}_{feat}"
        right_col = f"{right}_{feat}"

        if left_col in df.columns and right_col in df.columns:
            df[f"{left}_{right}_{feat}_diff"] = df[left_col] - df[right_col]
            df[f"{left}_{right}_{feat}_absdiff"] = (df[left_col] - df[right_col]).abs()

# 3. Previous-epoch context features
# We only use previous epoch, not next epoch, to avoid using future information.
context_features = []

for ch in CHANNELS:
    for feat in BANDS + ["mean", "std", "energy"]:
        col = f"{ch}_{feat}"
        if col in df.columns:
            context_features.append(col)

for col in context_features:
    df[f"prev_{col}"] = (
        df.groupby("subject_group")[col]
        .shift(1)
    )

# First epoch of each subject has no previous epoch, so fill it with current value
for col in context_features:
    prev_col = f"prev_{col}"
    df[prev_col] = df[prev_col].fillna(df[col])

# Clean possible infinity values from ratios
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

print("Enhanced shape:", df.shape)
print("New features added:", df.shape[1] - pd.read_csv(INPUT_FILE, nrows=1).shape[1])

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Saved:", OUTPUT_FILE)