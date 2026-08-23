"""
Integration — builds the unified wildfire training dataset.

Unit of analysis: commune × day (fire season, fire-prone communes only)

Design:
  - Full dense calendar is built first for all commune × fire-season days.
  - Modeling skeleton keeps all positive next-day fire days + 4× sampled negatives.
  - Target: next_day_risk_class (0=LOW, 1=MODERATE, 2=HIGH).
  - Same-day fire activity is retained as a valid predictor of next-day risk.
  - Dense calendar is saved separately for rolling, lag, anomaly, and baseline
    feature engineering before features are added to the modeling dataset.

Join strategy:
  FIRMS      → commune × day fire aggregates.
  ERA5       → nearest ERA5 cell to each commune centroid.
  Sentinel-2 → monthly commune-level vegetation indices.
  DEM/WorldPop/OSM → static commune-level zonal features.
  Land cover → commune-level WorldCover fractions.

Outputs:
  data/training/integrated/algeria_wildfire_dataset.parquet
  data/training/integrated/algeria_wildfire_dense_calendar.parquet
  data/training/integrated/commune_static_features.parquet
  data/training/integrated/algeria_wildfire_dataset_sample100.csv
"""

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.logger import setup_logger


# ── CONFIG ────────────────────────────────────────────────────────────────────
import yaml

with open("configs/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

CURATED = Path(CONFIG["gadm"]["paths"]["curated"]).parent   # data/training/curated
OUT_DIR = Path(CONFIG["training"]["integrated"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARIES_DIR = Path(CONFIG["gadm"]["paths"]["curated"])
DEM_DIR        = Path(CONFIG["dem"]["paths"]["curated"])
WORLDPOP_DIR   = Path(CONFIG["worldpop"]["paths"]["curated"])
ROADS_DIR      = Path(CONFIG["osm"]["paths"]["curated"])
ERA5_DIR       = Path(CONFIG["era5"]["paths"]["curated_training"])
SENTINEL_DIR   = Path(CONFIG["sentinel"]["paths"]["curated_training"])
FIRMS_DIR      = Path(CONFIG["firms"]["paths"]["curated_training"])
LANDCOVER_DIR  = Path(CONFIG["landcover"]["paths"]["curated"])

# Commune mask thresholds
MIN_FOREST_PCT       = 0.10  # % ESA WorldCover forest/shrubland to keep commune
MIN_HISTORICAL_FIRES = 1     # min FIRMS detections 2012-2024 to keep commune
SAHARAN_WILAYAS = [
    "DZA.1_1",   # Adrar
    "DZA.41_1",  # Tamanrasset  
    "DZA.22_1",  # Illizi     
    "DZA.31_1",  # Naâma        
    "DZA.17_1",  # El Bayadh    
    "DZA.44_1",  # Tindouf    
    "DZA.33_1",  # Ouargla     
    "DZA.20_1",  # Ghardaïa    
    "DZA.7_1",   # Béchar       
    "DZA.9_1",   # Biskra        
    "DZA.16_1",  # Djelfa        
    "DZA.25_1",  # Laghouat      
    "DZA.18_1",  # El Oued       
]

# Stratified sampling ratio (negatives per positive)
NEG_RATIO = 4

# Risk class thresholds — calibrate after first dataset build
# using: firms_agg.groupby(["date","commune_id"])["frp"].agg(["count","sum"]).describe()
MODERATE_FRP_THRESHOLD = 100.0  # MW — below this with low count -> MODERATE
SIGNIFICANT_COUNT      = 3      # detections — above this -> HIGH regardless of FRP


# ── RISK CLASS ASSIGNMENT ─────────────────────────────────────────────────────

def assign_risk_class(fire_count: pd.Series, frp_sum: pd.Series) -> pd.Series:
    """
    3-class risk label applied to NEXT-DAY FIRMS values.

    0 = LOW       — no detections tomorrow
    1 = MODERATE  — 1-2 detections and FRP < threshold (small/early fire)
    2 = HIGH      — 3+ detections or FRP >= threshold (significant event)

    Thresholds set conservatively for commune scale — even 1 VIIRS detection
    at commune level is meaningful. Calibrate after first build.
    """
    conditions = [
        fire_count == 0,
        (fire_count >= 1) & (fire_count < SIGNIFICANT_COUNT) & (frp_sum < MODERATE_FRP_THRESHOLD),
    ]
    choices = [0, 1]
    return np.select(conditions, choices, default=2)


RISK_LABELS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}


# ── STEP 0: COMMUNE FILTER MASK ───────────────────────────────────────────────

def load_fire_prone_communes(communes: gpd.GeoDataFrame,
                              firms_path: Path,
                              config: dict) -> gpd.GeoDataFrame:
    """
    Filter 1541 communes to fire-prone subset (~400-600 expected).

    Criteria (any one sufficient):
      1. Historical FIRMS fire count >= MIN_HISTORICAL_FIRES (2012-2024)
      2. ESA WorldCover forest/shrubland % >= MIN_FOREST_PCT (if available)

    Always excludes deep Saharan wilayas regardless of criteria.
    """
    print(f"\n{'='*55}")
    print(f"COMMUNE FILTER MASK")
    print(f"{'='*55}")
    print(f"Starting communes: {len(communes)}")

    # Exclude deep Saharan wilayas
    before = len(communes)
    communes = communes[~communes["wilaya_id"].isin(SAHARAN_WILAYAS)].copy()
    print(f"After Sahara exclusion: {len(communes)} (removed {before - len(communes)})")

    # Filter by historical fire presence
    try:
        firms = pd.read_parquet(firms_path)
        firms = firms.rename(columns={"GID_2": "commune_id", "GID_1": "wilaya_id"})
        firms["commune_id"] = firms["commune_id"].astype(str)

        fire_counts = (
            firms.groupby("commune_id")
            .size()
            .reset_index(name="historical_fire_count")
        )

        communes["commune_id"] = communes["commune_id"].astype(str)
        communes = communes.merge(fire_counts, on="commune_id", how="left")
        communes["historical_fire_count"] = communes["historical_fire_count"].fillna(0).astype(int)

        has_fires = communes["historical_fire_count"] >= MIN_HISTORICAL_FIRES
        print(f"Communes with >= {MIN_HISTORICAL_FIRES} historical fire detection(s): "
              f"{has_fires.sum()}")

    except Exception as e:
        print(f"  WARNING: Could not load FIRMS for commune filtering: {e}")
        print(f"  Falling back to Sahara exclusion only.")
        has_fires = pd.Series(True, index=communes.index)

    # Filter by land cover if available
    lc_files = sorted(LANDCOVER_DIR.glob("landcover_*.parquet"))

    if lc_files:
        try:
            lc = pd.read_parquet(lc_files[-1])
            lc["commune_id"] = lc["commune_id"].astype(str)

            # Compute burnable vegetation percentage
            if "burnable_fraction" in lc.columns:
                lc["forest_pct"] = lc["burnable_fraction"]
            elif "forest_fraction" in lc.columns:
                lc["forest_pct"] = (
                    lc["forest_fraction"] +
                    lc.get("shrub_fraction", 0)
                )
            else:
                lc["forest_pct"] = 0.0

            communes = communes.merge(
                lc[["commune_id", "forest_pct"]],
                on="commune_id",
                how="left"
            )

            communes["forest_pct"] = communes["forest_pct"].fillna(0.0)

            has_forest = communes["forest_pct"] >= MIN_FOREST_PCT

            print(
                f"Communes with >= {MIN_FOREST_PCT}% forest/shrubland: "
                f"{has_forest.sum()}"
            )

            communes = communes[has_fires | has_forest].copy()

        except Exception as e:
            print(f"  WARNING: Could not load land cover: {e}")
            communes = communes[has_fires].copy()

    else:
        communes = communes[has_fires].copy()

    print(f"\nFire-prone communes after filtering: {len(communes)}")
    print(f"Wilayas represented: {communes['wilaya_id'].nunique()}")
    return communes.reset_index(drop=True)


# ── STEP 1: AGGREGATE FIRMS TO COMMUNE × DAY ─────────────────────────────────

def aggregate_firms(firms_path: Path) -> pd.DataFrame:
    """
    Aggregate FIRMS detections to commune × day.
    Returns DataFrame: date, commune_id, fire_count, frp_sum, frp_max
    """
    firms = pd.read_parquet(firms_path)
    firms = firms.rename(columns={"GID_2": "commune_id", "GID_1": "wilaya_id"})
    firms["acq_date"]   = pd.to_datetime(firms["acq_date"])
    firms["date"]       = firms["acq_date"].dt.normalize()
    firms["commune_id"] = firms["commune_id"].astype(str)

    n_before    = len(firms)
    n_unmatched = firms["commune_id"].isna().sum()
    if n_unmatched:
        print(f"Dropping {n_unmatched:,} / {n_before:,} FIRMS detections "
              f"with no commune_id ({100*n_unmatched/n_before:.1f}%)")
    firms = firms.dropna(subset=["commune_id"])

    agg = (
        firms
        .groupby(["date", "commune_id"])
        .agg(
            fire_count=("frp", "count"),
            frp_sum   =("frp", "sum"),
            frp_max   =("frp", "max"),
        )
        .reset_index()
    )

    print(f"FIRMS aggregated: {len(agg):,} commune x day fire events "
          f"across {agg['commune_id'].nunique()} communes")
    return agg


# ── STEP 2: BUILD STRATIFIED SKELETON ────────────────────────────────────────

def build_stratified_skeleton(communes: gpd.GeoDataFrame,
                               firms_agg: pd.DataFrame,
                               start: str,
                               end: str,
                               fire_months: list,
                               neg_ratio: int = NEG_RATIO):
    """
    Returns (skeleton, full).

    `full` is the complete, unsampled calendar and must be used for all
    time-based feature engineering.

    `skeleton` is the target-conditioned sampled dataset and must not be used
    for rolling, lag, or baseline calculations.
    """
    print(f"\n{'='*55}")
    print(f"STRATIFIED SKELETON")
    print(f"{'='*55}")

    all_dates   = pd.date_range(start, end, freq="D")
    all_dates   = all_dates[all_dates.month.isin(fire_months)]
    commune_ids = communes["commune_id"].astype(str).values

    full = pd.MultiIndex.from_product(
        [all_dates, commune_ids], names=["date", "commune_id"]
    ).to_frame(index=False)

    full = full.merge(firms_agg, on=["date", "commune_id"], how="left")
    full["fire_count"] = full["fire_count"].fillna(0).astype(int)
    full["frp_sum"]    = full["frp_sum"].fillna(0.0)
    full["frp_max"]    = full["frp_max"].fillna(0.0)

    # ── NEXT-DAY TARGET — group by (commune_id, season_year) ─────────────
    # Keeps the next-day target within the same fire season and prevents
    # crossing the off-season gap (e.g. Oct 31 → May 1).
    # season_year is kept so all temporal features use the same grouping.
    full["season_year"] = full["date"].dt.year
    full = full.sort_values(["commune_id", "season_year", "date"])

    full["next_day_fire_count"] = (
        full.groupby(["commune_id", "season_year"])["fire_count"].shift(-1)
    )
    full["next_day_frp_sum"] = (
        full.groupby(["commune_id", "season_year"])["frp_sum"].shift(-1)
    )

    # Drop last day of EACH season — no next-day label available
    full = full.dropna(subset=["next_day_fire_count"]).copy()
    full["next_day_fire_count"] = full["next_day_fire_count"].astype(int)
    full["next_day_frp_sum"]    = full["next_day_frp_sum"].fillna(0.0)

    full["next_day_risk_class"] = assign_risk_class(
        full["next_day_fire_count"], full["next_day_frp_sum"]
    )
    full["next_day_risk_label"] = full["next_day_risk_class"].map(RISK_LABELS)

    positives         = full[full["next_day_risk_class"] > 0].copy()
    negatives         = full[full["next_day_risk_class"] == 0].copy()
    n_sample          = min(len(negatives), len(positives) * neg_ratio)
    negatives_sampled = negatives.sample(n=n_sample, random_state=42)

    skeleton = pd.concat(
        [positives, negatives_sampled]
    ).sort_values(["commune_id", "date"]).reset_index(drop=True)

    total  = len(skeleton)
    n_pos  = len(positives)
    n_mod  = (skeleton["next_day_risk_class"] == 1).sum()
    n_high = (skeleton["next_day_risk_class"] == 2).sum()
    n_low  = (skeleton["next_day_risk_class"] == 0).sum()

    print(f"Full grid before sampling: {len(full):,} rows")
    print(f"  Positives (next-day fire): {n_pos:,} ({100*n_pos/len(full):.1f}% of full grid)")
    print(f"\nStratified skeleton: {total:,} rows")
    print(f"  0 LOW:      {n_low:>8,} ({100*n_low/total:.1f}%)")
    print(f"  1 MODERATE: {n_mod:>8,} ({100*n_mod/total:.1f}%)")
    print(f"  2 HIGH:     {n_high:>8,} ({100*n_high/total:.1f}%)")
    print(f"  Positive rate: {100*(n_mod+n_high)/total:.1f}%")

    return skeleton, full


# ── STEP 3: JOIN ERA5 WEATHER ─────────────────────────────────────────────────

def map_communes_to_era5(communes: gpd.GeoDataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    """
    Maps each commune to its nearest ERA5 cell using a BallTree.
    Shared by both the modeling skeleton and dense calendar to ensure
    consistent ERA5 assignments.
    """
    from sklearn.neighbors import BallTree

    centroids = communes.to_crs("EPSG:32631").copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids = centroids.to_crs("EPSG:4326")
    centroids["centroid_lon"] = centroids.geometry.x
    centroids["centroid_lat"] = centroids.geometry.y

    era5_cells = (
        era5[["era5_cell_id", "latitude", "longitude"]]
        .drop_duplicates("era5_cell_id")
    )

    era5_coords     = np.radians(era5_cells[["latitude", "longitude"]].to_numpy())
    centroid_coords = np.radians(
        centroids[["centroid_lat", "centroid_lon"]].to_numpy()
    )

    tree   = BallTree(era5_coords, metric="haversine")
    _, idx = tree.query(centroid_coords, k=1)
    idx    = idx.ravel()

    commune_to_era5 = pd.DataFrame({
        "commune_id":   centroids["commune_id"].astype(str).values,
        "era5_cell_id": era5_cells.iloc[idx]["era5_cell_id"].values,
    })

    n_per_cell = commune_to_era5.groupby("era5_cell_id")["commune_id"].nunique()
    print(f"ERA5 cell sharing: median {n_per_cell.median():.0f} communes/cell, "
          f"max {n_per_cell.max()} sharing one cell")
    return commune_to_era5


def join_era5(df: pd.DataFrame,era5_path: Path,communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Join ERA5-Land weather to each commune x day row via nearest cell (BallTree).
    Many small communes map to the same ERA5 cell — expected at 9km resolution.
    """
    print(f"\n{'='*55}")
    print(f"ERA5 JOIN")
    print(f"{'='*55}")

    era5 = pd.read_parquet(era5_path)
    era5["date"] = pd.to_datetime(era5["date"])

    commune_to_era5 = map_communes_to_era5(communes, era5)

    df["commune_id"] = df["commune_id"].astype(str)
    df = df.merge(commune_to_era5, on="commune_id", how="left")

    era5_feature_cols = [
        "date", "era5_cell_id",
        "temp_c", "rh", "wind_speed_kmh", "wind_dir",
        "precip_mm", "soil_moisture",
        "FFMC", "DMC", "DC", "ISI", "BUI", "FWI",
    ]
    df = df.merge(era5[era5_feature_cols], on=["date", "era5_cell_id"], how="left")

    missing = df["temp_c"].isna().sum()
    print(f"ERA5 join complete: {missing:,} rows missing weather "
          f"({100*missing/len(df):.2f}%)")
    return df


def build_dense_calendar(full: pd.DataFrame,
                          era5_path: Path,
                          communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Builds the dense calendar used for temporal feature engineering.

    `full` contains every commune × fire-season day. This function adds daily
    weather and the same-day wilaya fire sum using all fire-prone communes.

    The dense calendar is used for rolling, lag, days-since-fire, anomaly, and
    baseline features; those features are computed later in the notebook.
    """
    print(f"\n{'='*55}")
    print(f"DENSE CALENDAR (for rolling / lag / baseline features)")
    print(f"{'='*55}")

    era5 = pd.read_parquet(era5_path)
    era5["date"] = pd.to_datetime(era5["date"])
    commune_to_era5 = map_communes_to_era5(communes, era5)

    dense = full[[
        "date", "commune_id", "season_year",
        "fire_count", "frp_sum", "next_day_risk_class",
    ]].copy()

    # Wilaya id needed for the same-day neighbor-fire feature
    commune_meta = communes[["commune_id", "wilaya_id"]].copy()
    commune_meta["commune_id"] = commune_meta["commune_id"].astype(str)
    dense["commune_id"] = dense["commune_id"].astype(str)
    dense = dense.merge(commune_meta, on="commune_id", how="left")

    # Same-day wilaya-wide fire sum over ALL fire-prone communes (dense) 
    wilaya_sum = (
        dense.groupby(["wilaya_id", "date"])["fire_count"]
        .sum().reset_index().rename(columns={"fire_count": "wilaya_fire_sum"})
    )
    dense = dense.merge(wilaya_sum, on=["wilaya_id", "date"], how="left")

    # Daily weather needed for rolling FWI/DC/DMC/precip and the FFMC baseline
    dense = dense.merge(commune_to_era5, on="commune_id", how="left")
    weather_cols = ["FWI", "DC", "DMC", "precip_mm", "FFMC"]
    dense = dense.merge(
        era5[["date", "era5_cell_id"] + weather_cols],
        on=["date", "era5_cell_id"], how="left"
    )

    missing = dense["FWI"].isna().sum()
    print(f"Dense calendar built: {len(dense):,} rows "
          f"({dense['commune_id'].nunique()} communes), "
          f"{missing:,} rows missing weather ({100*missing/len(dense):.2f}%)")
    return dense


# ── STEP 4: JOIN SENTINEL-2 VEGETATION INDICES ────────────────────────────────

def join_sentinel(df: pd.DataFrame, sentinel_path: Path) -> pd.DataFrame:
    """
    Join Sentinel-2 monthly indices on (commune_id, year, month).
    Monthly resolution is correct — vegetation dryness changes slowly.
    Every day in August gets the same August NDVI — intentional.
    """
    print(f"\n{'='*55}")
    print(f"SENTINEL-2 JOIN")
    print(f"{'='*55}")

    sentinel = pd.read_parquet(sentinel_path)
    sentinel["date"]       = pd.to_datetime(sentinel["date"])
    sentinel["year"]       = sentinel["date"].dt.year
    sentinel["month"]      = sentinel["date"].dt.month
    sentinel["commune_id"] = sentinel["commune_id"].astype(str)

    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df = df.merge(
        sentinel[["commune_id", "year", "month", "NDVI", "NDWI", "NBR"]],
        on=["commune_id", "year", "month"],
        how="left",
    )

    missing = df["NDVI"].isna().sum()
    print(f"Sentinel join complete: {missing:,} rows missing NDVI "
          f"({100*missing/len(df):.2f}%)")
    return df


# ── STEP 5: JOIN STATIC RASTERS (cached) ──────────────────────────────────────

def compute_static_features(communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute zonal stats for static rasters. Slow — cached after first run."""
    from rasterstats import zonal_stats

    static_layers = [
        (DEM_DIR      / "elevation.tif",          "elevation_mean_m",      "mean"),
        (DEM_DIR      / "slope.tif",               "slope_mean_deg",        "mean"),
        (DEM_DIR      / "aspect.tif",              "aspect_mean_deg",       "mean"),
        (WORLDPOP_DIR / "population_density.tif",  "pop_density_mean",      "mean"),
        (ROADS_DIR    / "road_distance.tif",       "road_distance_mean_km", "mean"),
    ]

    static_df = pd.DataFrame({"commune_id": communes["commune_id"].astype(str).values})

    for tif_path, col_name, stat in static_layers:
        if not tif_path.exists():
            print(f"  Missing raster: {tif_path.name} — filling NaN")
            static_df[col_name] = float("nan")
            continue
        results = zonal_stats(
            communes,
            str(tif_path),
            stats=[stat],
            nodata=float("nan"),
            all_touched=True,
        )
        static_df[col_name] = [r[stat] for r in results]
        print(f"  Done: {col_name}")

    return static_df


def join_static_rasters(df: pd.DataFrame,
                         communes: gpd.GeoDataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"STATIC FEATURES JOIN")
    print(f"{'='*55}")

    static_cache     = OUT_DIR / "commune_static_features.parquet"
    lc_cols_expected = ["forest_fraction", "burnable_fraction"]

    # Check if cache exists AND has landcover columns
    need_rebuild = True
    if static_cache.exists():
        _check = pd.read_parquet(static_cache)
        if all(c in _check.columns for c in lc_cols_expected):
            need_rebuild = False
            print(f"Loading cached static features from {static_cache.name}")
            static_df = _check
        else:
            print("Static cache missing landcover columns — rebuilding...")
            static_cache.unlink()

    if need_rebuild:
        print("Computing zonal stats (first run or rebuild — will be cached)...")
        static_df = compute_static_features(communes)

        lc_files = sorted(LANDCOVER_DIR.glob("landcover_*.parquet"))
        if lc_files:
            lc = pd.read_parquet(lc_files[-1])
            lc["commune_id"] = lc["commune_id"].astype(str)
            lc_cols = ["commune_id", "forest_fraction", "shrub_fraction",
                       "grass_fraction", "crop_fraction", "burnable_fraction"]
            lc_cols = [c for c in lc_cols if c in lc.columns]
            static_df = static_df.merge(lc[lc_cols], on="commune_id", how="left")
            print(f"  Landcover fractions joined: {lc['commune_id'].nunique()} communes")
        else:
            print("  WARNING: No landcover parquet found — run landcover.py curate() first")
            for col in ["forest_fraction", "shrub_fraction",
                        "grass_fraction", "crop_fraction", "burnable_fraction"]:
                static_df[col] = float("nan")

        static_df.to_parquet(static_cache, index=False)
        print(f"Cached -> {static_cache}")

    df["commune_id"]        = df["commune_id"].astype(str)
    static_df["commune_id"] = static_df["commune_id"].astype(str)
    df = df.merge(static_df, on="commune_id", how="left")
    print(f"Static join complete")
    return df


# ── STEP 6: ADD TEMPORAL FEATURES ────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame, fire_months: list) -> pd.DataFrame:
    df["day_of_year"]   = df["date"].dt.dayofyear
    season_start_month  = min(fire_months)
    df["day_of_season"] = (
        df["date"] - pd.to_datetime(
            df["date"].dt.year.astype(str) + f"-{season_start_month:02d}-01"
        )
    ).dt.days.clip(lower=0)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


# ── STEP 7: FINAL CLEANUP AND SAVE ───────────────────────────────────────────

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Final column ordering, NaN fill for cross-source gaps, quality report."""

    # ── Compute mean fire intensity before column selection ───────────────
    # frp_sum and frp_max dropped from feature_cols below — mean_frp replaces them
    df["mean_frp"] = np.where(
        df["fire_count"] > 0,
        df["frp_sum"] / df["fire_count"].clip(lower=1),
        0.0
    ).round(2)

    feature_cols = [
        # Identifiers
        "date", "commune_id", "commune_name", "wilaya_id", "wilaya_name",
        # Temporal
        "year", "month", "day_of_year", "day_of_season", "week_of_year",
        # Same-day fire state (valid feature, not leakage)
        "fire_count", "mean_frp",
        # frp_sum and frp_max intentionally excluded — replaced by mean_frp
        # Weather (ERA5-Land)
        "temp_c", "rh", "wind_speed_kmh", "wind_dir",
        "precip_mm", "soil_moisture",
        # Fire Weather Index
        "FFMC", "DMC", "DC", "ISI", "BUI", "FWI",
        # Vegetation (Sentinel-2, monthly)
        "NDVI", "NDWI", "NBR",
        # Terrain
        "elevation_mean_m", "slope_mean_deg", "aspect_mean_deg",
        # Human exposure
        "pop_density_mean", "road_distance_mean_km",
        # Land cover
        "forest_fraction", "shrub_fraction", "grass_fraction",
        "crop_fraction", "burnable_fraction",
        # ERA5 cell reference (debugging/XAI)
        "era5_cell_id",
        # TARGET
        "next_day_fire_count", "next_day_frp_sum",
        "next_day_risk_class", "next_day_risk_label",
    ]

    df = df[[c for c in feature_cols if c in df.columns]].copy()

    weather_cols  = ["temp_c", "rh", "wind_speed_kmh", "wind_dir",
                     "precip_mm", "soil_moisture",
                     "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
    sentinel_cols = ["NDVI", "NDWI", "NBR"]
    static_cols   = ["elevation_mean_m", "slope_mean_deg", "aspect_mean_deg",
                     "pop_density_mean", "road_distance_mean_km",
                     "forest_fraction", "shrub_fraction", "grass_fraction",
                     "crop_fraction", "burnable_fraction"]

    # ── Weather NaN fill — sort by date within each ERA5 cell first ───────
    # After stratified sampling + merges, row order is arbitrary.
    # ffill/bfill must operate on chronologically ordered rows.
    df = df.sort_values(["era5_cell_id", "date"])
    for col in weather_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df.groupby("era5_cell_id")[col].transform(
                lambda x: x.ffill().bfill()
            )

    # ── Sentinel NaN fill — sort by date within each commune first ────────
    df = df.sort_values(["commune_id", "date"])
    for col in sentinel_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df.groupby("commune_id")[col].transform(
                lambda x: x.ffill().bfill()
            )

    # ── Static NaN fill — wilaya median (terrain varies slowly) ──────────
    for col in static_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df.groupby("wilaya_id")[col].transform(
                lambda x: x.fillna(x.median())
            )

    # ── Hard drop — rows still missing target or core weather ─────────────
    before = len(df)
    df = df.dropna(subset=["next_day_risk_class", "temp_c", "FWI"])
    if len(df) < before:
        print(f"Dropped {before - len(df):,} rows still missing "
              f"target or core weather after all fills")

    print(f"\n{'='*55}")
    print(f"FINAL DATASET PROFILE")
    print(f"{'='*55}")
    print(f"Shape:      {df.shape}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Communes:   {df['commune_id'].nunique()}")
    print(f"Wilayas:    {df['wilaya_id'].nunique()}")

    print(f"\nTarget distribution (next_day_risk_class):")
    dist = df["next_day_risk_class"].value_counts().sort_index()
    for cls, count in dist.items():
        pct = 100 * count / len(df)
        print(f"  {cls} ({RISK_LABELS[cls]:<8}): {count:>8,} rows ({pct:.1f}%)")

    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  None")
    else:
        for col, n in missing.items():
            print(f"  {col}: {n:,} ({100*n/len(df):.1f}%)")

    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    logger = setup_logger()

    fire_months = CONFIG["training"]["fire_season_months"]
    start_date  = CONFIG["training"]["start_date"]
    end_date    = CONFIG["training"]["end_date"]

    print(f"\n{'='*55}")
    print(f"  Integration — building unified training dataset")
    print(f"  Period:      {start_date} -> {end_date}")
    print(f"  Fire months: {fire_months}")
    print(f"{'='*55}")

    # Load communes
    communes_path = BOUNDARIES_DIR / "algeria_communes.gpkg"
    communes = gpd.read_file(communes_path).to_crs("EPSG:4326")
    communes = communes.rename(columns={
        "GID_1": "wilaya_id",
        "NAME_1": "wilaya_name",
        "GID_2": "commune_id",
        "NAME_2": "commune_name",
    })
    print(f"\nAll communes loaded: {len(communes)}")

    # Resolve source files
    firms_files    = sorted(FIRMS_DIR.glob("*.parquet"))
    era5_files     = sorted(ERA5_DIR.glob("*.parquet"))
    sentinel_files = sorted(SENTINEL_DIR.glob("*.parquet"))

    if not firms_files:
        raise FileNotFoundError("No FIRMS parquet — run firms.py curate() first")
    if not era5_files:
        raise FileNotFoundError("No ERA5 parquet — run era5.py curate() first")

    firms_path    = firms_files[-1]
    era5_path     = era5_files[-1]
    sentinel_path = sentinel_files[-1] if sentinel_files else None

    print(f"\nSource files:")
    print(f"  FIRMS:    {firms_path.name}")
    print(f"  ERA5:     {era5_path.name}")
    print(f"  Sentinel: {sentinel_path.name if sentinel_path else 'NOT FOUND — skipping'}")

    # Step 0: Filter to fire-prone communes
    communes = load_fire_prone_communes(communes, firms_path, CONFIG)

    # Step 1: Aggregate FIRMS to commune x day
    firms_agg = aggregate_firms(firms_path)

    # Filter firms_agg to fire-prone communes only
    fire_prone_ids = set(communes["commune_id"].astype(str).values)
    firms_agg = firms_agg[firms_agg["commune_id"].isin(fire_prone_ids)].copy()

    # Step 2: Build stratified skeleton (+ dense calendar for rolling features)
    df, full_grid = build_stratified_skeleton(
        communes, firms_agg, start_date, end_date,
        fire_months, neg_ratio=NEG_RATIO
    )

    dense_calendar = build_dense_calendar(full_grid, era5_path, communes)
    dense_path = OUT_DIR / "algeria_wildfire_dense_calendar.parquet"
    dense_calendar.to_parquet(dense_path, index=False)
    print(f"\nSaved dense calendar -> {dense_path}")
    print(f"   {dense_calendar.shape[0]:,} rows x {dense_calendar.shape[1]} columns")

    # Step 3: Join commune metadata
    commune_meta = communes[
        ["commune_id", "commune_name", "wilaya_id", "wilaya_name"]
    ].copy()
    commune_meta["commune_id"] = commune_meta["commune_id"].astype(str)
    df["commune_id"] = df["commune_id"].astype(str)
    df = df.merge(commune_meta, on="commune_id", how="left")

    # Step 4: Add temporal features
    df = add_temporal_features(df, fire_months)

    # Step 5: Join ERA5 weather
    df = join_era5(df, era5_path, communes)

    # Step 6: Join Sentinel-2 (optional)
    if sentinel_path:
        df = join_sentinel(df, sentinel_path)
    else:
        print("\nSentinel-2 skipped — NDVI/NDWI/NBR will be NaN")
        df["NDVI"] = float("nan")
        df["NDWI"] = float("nan")
        df["NBR"]  = float("nan")

    # Step 7: Join static rasters
    df = join_static_rasters(df, communes)

    # Step 8: Finalize
    df = finalize(df)

    # Save
    out_path = OUT_DIR / "algeria_wildfire_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"   {df.shape[0]:,} rows x {df.shape[1]} columns")

    sample_path = OUT_DIR / "algeria_wildfire_dataset_sample100.csv"
    df.sample(min(100, len(df)), random_state=42).to_csv(sample_path, index=False)
    print(f"   Sample -> {sample_path.name}")


if __name__ == "__main__":
    main()