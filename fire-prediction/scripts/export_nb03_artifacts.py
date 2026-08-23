import pandas as pd
from pathlib import Path

# Setup Directory Paths
INTEGRATED_PATH = Path("data/training/integrated/algeria_wildfire_dataset.parquet")
DENSE_PATH = Path("data/training/integrated/algeria_wildfire_dense_calendar.parquet")
ENGINEERED_DIR = Path("data/training/engineered")
ENGINEERED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = [2015, 2016, 2017, 2018]

print("Loading integrated and dense datasets...")
df = pd.read_parquet(INTEGRATED_PATH)
dense = pd.read_parquet(DENSE_PATH)

# Ensure proper datetime parsing
df["date"] = pd.to_datetime(df["date"])
dense["date"] = pd.to_datetime(dense["date"])
dense["month"] = dense["date"].dt.month

# 1. Commune Fire Rate Artifact
print("Generating commune_fire_rate.parquet...")
dense_train_mask = dense["season_year"].isin(TRAIN_YEARS)
train_fire_rate = (
    dense[dense_train_mask]
    .groupby("commune_id")["next_day_risk_class"]
    .apply(lambda x: (x > 0).mean())
    .reset_index()
    .rename(columns={"next_day_risk_class": "commune_fire_rate"})
)
train_fire_rate.to_parquet(ENGINEERED_DIR / "commune_fire_rate.parquet", index=False)

# 2. FFMC Baseline Artifact
print("Generating anomaly_baselines.parquet (FFMC)...")
dense_train = dense[dense["season_year"].isin(TRAIN_YEARS)]
ffmc_baseline = (
    dense_train.groupby(["commune_id", "month"])["FFMC"]
    .mean()
    .reset_index()
    .rename(columns={"FFMC": "FFMC_baseline"})
)
ffmc_baseline.to_parquet(ENGINEERED_DIR / "anomaly_baselines.parquet", index=False)

# 3. NBR Baseline Artifact
print("Generating nbr_baselines.parquet...")
train_ref = df[df["year"].isin(TRAIN_YEARS)].copy()
nbr_baseline = (
    train_ref.groupby(["commune_id", "month"])["NBR"]
    .mean()
    .reset_index()
    .rename(columns={"NBR": "NBR_baseline"})
)
nbr_baseline.to_parquet(ENGINEERED_DIR / "nbr_baselines.parquet", index=False)

# 4. ERA5 Spatial Mapping Artifact
print("Generating commune_era5_mapping.parquet...")
era5_mapping = (
    df[["commune_id", "era5_cell_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
era5_mapping.to_parquet(ENGINEERED_DIR / "commune_era5_mapping.parquet", index=False)

print("\nSuccess: All 4 artifact files successfully generated in data/training/engineered/")