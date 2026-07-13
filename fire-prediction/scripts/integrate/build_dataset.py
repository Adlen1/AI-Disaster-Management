# scripts/integrate/build_dataset.py
"""
Integration — builds the unified training dataset.

Unit of analysis: wilaya × day (fire season only, 2015–2025)

Join strategy:
  FIRMS    → wilaya × day fire label (already has wilaya_id from spatial join)
  ERA5     → nearest ERA5 cell to each wilaya centroid → weather per wilaya per day
  Sentinel → wilaya × month NDVI/NDWI/NBR (monthly, broadcast to all days in month)
  DEM      → mean elevation/slope/aspect per wilaya (static, join once)
  WorldPop → mean population density per wilaya (static, join once)
  OSM      → mean road distance per wilaya (static, join once)
  GADM     → wilaya names (static)

Output:
  data/integrated/algeria_wildfire_dataset.parquet
  → one row per (wilaya × day), ~430,000 rows for 2015-2025 fire seasons
  → target: fire_risk_class (0=none, 1=low, 2=moderate, 3=high, 4=critical)
"""

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.logger import setup_logger


# ── CONFIG ────────────────────────────────────────────────────────────────────

CURATED = Path("data/curated")
OUT_DIR = Path("data/integrated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARIES_DIR = CURATED / "boundaries"
DEM_DIR        = CURATED / "dem"
WORLDPOP_DIR   = CURATED / "worldpop"
ROADS_DIR      = CURATED / "roads"
ERA5_DIR       = CURATED / "era5"
SENTINEL_DIR   = CURATED / "sentinel"
FIRMS_DIR      = CURATED / "firms"


# ── RISK CLASS ASSIGNMENT ─────────────────────────────────────────────────────

def assign_risk_class(fire_count: pd.Series, frp_sum: pd.Series) -> pd.Series:
    """
    Convert raw FIRMS detection counts and FRP into a 5-class risk label.

    Classes:
      0 = NO_FIRE    — no detections
      1 = LOW        — 1-2 detections, low intensity
      2 = MODERATE   — small cluster or moderate intensity
      3 = HIGH       — significant fire activity
      4 = CRITICAL   — major fire event

    Thresholds derived from Algeria historical fire seasons.
    Revisit after first model training and adjust based on class balance.
    """
    conditions = [
        fire_count == 0,
        (fire_count <= 2)  & (frp_sum < 50),
        (fire_count <= 5)  & (frp_sum < 150),
        (fire_count <= 15) & (frp_sum < 500),
    ]
    choices = [0, 1, 2, 3]
    return np.select(conditions, choices, default=4)


RISK_LABELS = {
    0: "NO_FIRE",
    1: "LOW",
    2: "MODERATE",
    3: "HIGH",
    4: "CRITICAL"
}


# ── STEP 1: BUILD WILAYA × DAY SKELETON ──────────────────────────────────────

def build_skeleton(wilayas: gpd.GeoDataFrame,
                   start: str, end: str,
                   fire_months: list) -> pd.DataFrame:
    """
    Build the full (wilaya × day) grid for fire-season days only.
    This is the backbone every other source joins onto.
    """
    dates = pd.date_range(start, end, freq="D")
    dates = dates[dates.month.isin(fire_months)]

    wilaya_ids   = wilayas["wilaya_id"].values
    wilaya_names = wilayas["NAME_1"].values if "NAME_1" in wilayas.columns else wilaya_ids

    rows = []
    for date in dates:
        for wid, wname in zip(wilaya_ids, wilaya_names):
            rows.append({
                "date":        date,
                "wilaya_id":   wid,
                "wilaya_name": wname,
                "month":       date.month,
                "year":        date.year,
                "day_of_year": date.dayofyear,
            })

    df = pd.DataFrame(rows)
    print(f"Skeleton: {len(df):,} rows ({df['date'].nunique()} days × {len(wilaya_ids)} wilayas)")
    return df


# ── STEP 2: JOIN FIRMS FIRE LABELS ───────────────────────────────────────────

def join_firms(skeleton: pd.DataFrame, firms_path: Path) -> pd.DataFrame:
    """
    Aggregate FIRMS detections to wilaya × day level.
    Assign risk class based on detection count + FRP.
    """
    firms = pd.read_parquet(firms_path)
    firms["acq_date"] = pd.to_datetime(firms["acq_date"])
    firms["date"]     = firms["acq_date"].dt.normalize()

    # Aggregate: count detections and sum FRP per wilaya per day
    agg = (firms
           .groupby(["date", "wilaya_id"])
           .agg(
               fire_count=("frp", "count"),
               frp_sum=("frp", "sum"),
               frp_max=("frp", "max"),
           )
           .reset_index())

    # Join onto skeleton
    df = skeleton.merge(agg, on=["date", "wilaya_id"], how="left")
    df["fire_count"] = df["fire_count"].fillna(0).astype(int)
    df["frp_sum"]    = df["frp_sum"].fillna(0.0)
    df["frp_max"]    = df["frp_max"].fillna(0.0)

    # Assign risk class
    df["fire_risk_class"] = assign_risk_class(df["fire_count"], df["frp_sum"])
    df["fire_risk_label"] = df["fire_risk_class"].map(RISK_LABELS)

    # Log class distribution
    dist = df["fire_risk_class"].value_counts().sort_index()
    print(f"\nFire risk class distribution:")
    for cls, count in dist.items():
        pct = 100 * count / len(df)
        print(f"  {cls} ({RISK_LABELS[cls]:<10}): {count:>8,} rows ({pct:.1f}%)")

    return df


# ── STEP 3: JOIN ERA5 WEATHER ─────────────────────────────────────────────────

def join_era5(df: pd.DataFrame,
              era5_path: Path,
              wilayas: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Join ERA5 weather to each wilaya × day row.

    Strategy: for each wilaya, find the nearest ERA5 cell to the wilaya
    centroid. ERA5 is at ~31km so each wilaya maps to 1-3 ERA5 cells.
    Use the single nearest cell for simplicity (mean of nearby cells
    is a future improvement).
    """
    era5 = pd.read_parquet(era5_path)
    era5["date"] = pd.to_datetime(era5["date"])

    # Build wilaya centroid → nearest ERA5 cell mapping
    centroids = wilayas.to_crs("EPSG:32631")

    centroids["geometry"] = centroids.geometry.centroid

    centroids = centroids.to_crs("EPSG:4326")

    centroids["centroid_lon"] = centroids.geometry.x
    centroids["centroid_lat"] = centroids.geometry.y

    # Get unique ERA5 cell locations from one day's data
    era5_cells = era5[era5["date"] == era5["date"].min()][
        ["era5_cell_id", "latitude", "longitude"]
    ].drop_duplicates()

    # For each wilaya centroid, find nearest ERA5 cell
    from scipy.spatial import cKDTree
    tree = cKDTree(era5_cells[["latitude", "longitude"]].values)
    _, idx = tree.query(centroids[["centroid_lat", "centroid_lon"]].values)

    wilaya_to_era5 = pd.DataFrame({
        "wilaya_id":    centroids["wilaya_id"].values,
        "era5_cell_id": era5_cells.iloc[idx]["era5_cell_id"].values,
    })

    # Merge ERA5 cell ID onto skeleton
    df = df.merge(wilaya_to_era5, on="wilaya_id", how="left")

    # Join ERA5 weather by (date, era5_cell_id)
    era5_cols = ["date", "era5_cell_id", "temp_c", "rh", "wind_speed_kmh",
                 "wind_dir", "precip_mm", "soil_moisture",
                 "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
    df = df.merge(era5[era5_cols], on=["date", "era5_cell_id"], how="left")

    missing = df["temp_c"].isna().sum()
    print(f"\nERA5 join: {missing:,} rows with missing weather "
          f"({100*missing/len(df):.1f}%)")
    return df


# ── STEP 4: JOIN SENTINEL-2 VEGETATION INDICES ────────────────────────────────

def join_sentinel(df: pd.DataFrame, sentinel_path: Path) -> pd.DataFrame:
    """
    Join Sentinel-2 monthly indices to each wilaya × day row.
    Sentinel is monthly — broadcast each month's value to all days in that month.
    wilaya_id must match between FIRMS spatial join and Sentinel GEE regions.
    """
    sentinel = pd.read_parquet(sentinel_path)
    sentinel["date"]  = pd.to_datetime(sentinel["date"])
    sentinel["year"]  = sentinel["date"].dt.year
    sentinel["month"] = sentinel["date"].dt.month

    # Join on wilaya_id + year + month
    df = df.merge(
        sentinel[["wilaya_id", "year", "month", "NDVI", "NDWI", "NBR"]],
        on=["wilaya_id", "year", "month"],
        how="left"
    )

    missing = df["NDVI"].isna().sum()
    print(f"\nSentinel join: {missing:,} rows missing NDVI "
          f"({100*missing/len(df):.1f}%)")
    return df


# ── STEP 5: JOIN STATIC RASTER LAYERS ────────────────────────────────────────

def summarize_raster_per_wilaya(tif_path: Path,
                                wilayas: gpd.GeoDataFrame,
                                col_name: str,
                                stat: str = "mean") -> pd.DataFrame:
    """
    Compute summary statistic of a raster per wilaya polygon.
    Returns a small dataframe with wilaya_id and the summary value.
    """
    from rasterstats import zonal_stats

    results = zonal_stats(
        wilayas,
        str(tif_path),
        stats=[stat],
        nodata=float("nan"),
        all_touched=True,
    )

    return pd.DataFrame({
        "wilaya_id": wilayas["wilaya_id"].values,
        col_name:    [r[stat] for r in results],
    })


def join_static_rasters(df: pd.DataFrame,
                        wilayas: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Summarize DEM, WorldPop, and OSM rasters per wilaya.
    Join as static features (same value for every day in that wilaya).
    """
    print("\nSummarizing static rasters per wilaya...")

    static_layers = [
        (DEM_DIR     / "elevation.tif",        "elevation_mean_m",    "mean"),
        (DEM_DIR     / "slope.tif",             "slope_mean_deg",      "mean"),
        (DEM_DIR     / "aspect.tif",            "aspect_mean_deg",     "mean"),
        (WORLDPOP_DIR/ "population_density.tif","pop_density_mean",    "mean"),
        (ROADS_DIR   / "road_distance.tif",     "road_distance_mean_km","mean"),
    ]

    static_df = pd.DataFrame({"wilaya_id": wilayas["wilaya_id"].values})

    for tif_path, col_name, stat in static_layers:
        if not tif_path.exists():
            print(f"   Missing: {tif_path.name} — skipping")
            static_df[col_name] = float("nan")
            continue
        layer = summarize_raster_per_wilaya(tif_path, wilayas, col_name, stat)
        static_df = static_df.merge(layer, on="wilaya_id", how="left")
        print(f"  Success {col_name}")

    df = df.merge(static_df, on="wilaya_id", how="left")
    return df


# ── STEP 6: FINAL CLEANUP AND SAVE ───────────────────────────────────────────

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Final column ordering, type cleanup, and quality report."""

    # Canonical column order
    feature_cols = [
        # Identifiers
        "date", "wilaya_id", "wilaya_name", "year", "month", "day_of_year",
        # Weather (ERA5)
        "temp_c", "rh", "wind_speed_kmh", "wind_dir",
        "precip_mm", "soil_moisture",
        # Fire Weather Index
        "FFMC", "DMC", "DC", "ISI", "BUI", "FWI",
        # Vegetation (Sentinel-2)
        "NDVI", "NDWI", "NBR",
        # Terrain (DEM)
        "elevation_mean_m", "slope_mean_deg", "aspect_mean_deg",
        # Human exposure
        "pop_density_mean", "road_distance_mean_km",
        # ERA5 cell (for reference/debugging)
        "era5_cell_id",
        # Fire labels
        "fire_count", "frp_sum", "frp_max",
        "fire_risk_class", "fire_risk_label",
    ]

    # Keep only columns that exist
    df = df[[c for c in feature_cols if c in df.columns]].copy()

    # Impute missing NDVI/NDWI/NBR 
    df = impute_vegetation_gaps(df)

    print(f"\n{'='*55}")
    print(f"FINAL DATASET PROFILE")
    print(f"{'='*55}")
    print(f"Shape:         {df.shape}")
    print(f"Date range:    {df['date'].min()} → {df['date'].max()}")
    print(f"Wilayas:       {df['wilaya_id'].nunique()}")
    print(f"Fire days:     {(df['fire_risk_class'] > 0).sum():,} "
          f"({100*(df['fire_risk_class']>0).mean():.1f}%)")

    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  None")
    else:
        for col, n in missing.items():
            print(f"  {col}: {n:,} ({100*n/len(df):.1f}%)")

    return df

def impute_vegetation_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """ 
        Fill missing NDVI/NDWI/NBR by carrying forward/backward from the
        nearest month WITH data, per wilaya
    """
    df = df.sort_values(["wilaya_id", "date"]).copy()
 
    for col in ["NDVI", "NDWI", "NBR"]:
        before = df[col].isna().sum()
        # ffill/bfill operate per wilaya group, walking across months in
        # date order — this pulls from the nearest month that actually has
        # a value, forward first then backward for any still-missing gaps
        # at the very start of a wilaya's series.
        df[col] = df.groupby("wilaya_id")[col].transform(
            lambda s: s.ffill().bfill()
        )
        after = df[col].isna().sum()
        print(f"  {col}: imputed {before - after:,} / {before:,} missing values "
              f"({after:,} still missing — wilaya has NO data in the entire range)")
 
    return df

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import yaml
    logger = setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    fire_months = config["training"]["fire_season_months"]
    start_date  = config["training"]["start_date"]
    end_date    = config["training"]["end_date"]

    print(f"\n{'='*55}")
    print(f"  Integration — building unified training dataset")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Fire months: {fire_months}")
    print(f"{'='*55}\n")

    # Load wilayas — needed by multiple steps
    wilayas_path = BOUNDARIES_DIR / "algeria_wilayas.gpkg"
    wilayas = gpd.read_file(wilayas_path).to_crs("EPSG:4326")

    wilayas = wilayas.rename(columns={
        "GID_1": "wilaya_id"
    })
    print(f"Wilayas loaded: {len(wilayas)}")

    # Load curated source files
    firms_files    = sorted(FIRMS_DIR.glob("*.parquet"))
    era5_files     = sorted(ERA5_DIR.glob("*.parquet"))
    sentinel_files = sorted(SENTINEL_DIR.glob("*.parquet"))

    if not firms_files:
        raise FileNotFoundError("No FIRMS parquet — run firms.py curate() first")
    if not era5_files:
        raise FileNotFoundError("No ERA5 parquet — run era5.py curate() first")
    if not sentinel_files:
        raise FileNotFoundError("No Sentinel parquet — run sentinel.py ingest/curate() first")

    firms_path    = firms_files[-1]
    era5_path     = era5_files[-1]
    sentinel_path = sentinel_files[-1]

    print(f"\nSource files:")
    print(f"  FIRMS:    {firms_path.name}")
    print(f"  ERA5:     {era5_path.name}")
    print(f"  Sentinel: {sentinel_path.name}")

    # Build pipeline
    df = build_skeleton(wilayas, start_date, end_date, fire_months)
    df = join_firms(df, firms_path)
    df = join_era5(df, era5_path, wilayas)
    df = join_sentinel(df, sentinel_path)
    df = join_static_rasters(df, wilayas)
    df = finalize(df)

    # Save
    out_path = OUT_DIR / "algeria_wildfire_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"   {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Also save a CSV sample for quick inspection
    sample_path = OUT_DIR / "algeria_wildfire_dataset_sample100.csv"
    df.sample(min(100, len(df))).to_csv(sample_path, index=False)
    print(f"   Sample saved → {sample_path.name}")


if __name__ == "__main__":
    main()