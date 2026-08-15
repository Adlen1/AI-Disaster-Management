"""
NASA FIRMS (Fire archive + NRT) — Dynamic Data Source

Output files:
    curated/firms/firms_curated.parquet
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource
from dotenv import load_dotenv
from utils.base import filter_fire_season
import geopandas as gpd

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# Standard FIRMS column schema — same across all sensors
FIRMS_KEEP_COLS = [
    "latitude", "longitude", "acq_date", "acq_time",
    "satellite", "instrument", "confidence", "frp",
    "daynight", "type", "source_file"
]


class FIRMSSource(DataSource):

    BASE_URL      = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    MAX_DAYS_ARCH = 5   # max days per API call for archive sensors (_SP)

    def ingest(self, start_date: str = None, end_date: str = None):
        """
        - Download FIRMS data for Algeria via API.
        - Skip this if data already in raw/firms/ as CSV.
        """
        map_key = os.getenv("FIRMS_MAP_KEY")
        if not map_key:
            raise ValueError(
                "FIRMS_MAP_KEY not found in .env\n"
                "Get yours at: https://firms.modaps.eosdis.nasa.gov/api/map_key/"
            )

        bbox_list = self.config["algeria"]["bbox"]
        bbox = ",".join(str(x) for x in bbox_list)

        sensors = self.config["firms"]["sensors"]

        if not start_date:
            start_date = datetime.today().strftime("%Y-%m-%d")
        if not end_date:
            end_date = start_date

        start      = datetime.strptime(start_date, "%Y-%m-%d")
        end        = datetime.strptime(end_date,   "%Y-%m-%d")
        total_days = (end - start).days + 1

        self.logger.info(
            f"FIRMS ingestion: {start_date} → {end_date} "
            f"({total_days} days, {len(sensors)} sensors)"
        )

        existing = list(self.raw_dir.glob("*.csv"))
        if existing:
            self.logger.info(
                f"Found {len(existing)} existing CSV(s) in data/raw/firms/ — "
                f"skipping API ingestion. Delete raw files to force re-download."
            )
            return

        fire_months = self.config.get("training", {}).get(
            "fire_season_months", list(range(1, 13))
        )

        for sensor in sensors:
            all_dfs     = []
            chunk_start = start

            with tqdm(total=total_days, desc=f"{sensor}", unit="days") as pbar:
                while chunk_start <= end:

                    if chunk_start.month not in fire_months:
                        if chunk_start.month == 12:
                            next_month = chunk_start.replace(year=chunk_start.year + 1, month=1, day=1)
                        else:
                            next_month = chunk_start.replace(month=chunk_start.month + 1, day=1)
                        skipped = (next_month - chunk_start).days
                        pbar.update(skipped)
                        chunk_start = next_month
                        continue

                    chunk_end = min(
                        chunk_start + timedelta(days=self.MAX_DAYS_ARCH - 1),
                        end
                    )
                    n_days   = (chunk_end - chunk_start).days + 1
                    date_str = chunk_start.strftime("%Y-%m-%d")
                    url      = f"{self.BASE_URL}/{map_key}/{sensor}/{bbox}/{n_days}/{date_str}"

                    try:
                        df = pd.read_csv(url)
                        if not df.empty:
                            df["source_file"] = f"api_{sensor}_{date_str}"
                            all_dfs.append(df)
                            self.logger.debug(f"{date_str} +{n_days}d → {len(df)} rows")
                    except Exception as e:
                        self.logger.warning(f"Failed {date_str}: {e}")

                    pbar.update(n_days)
                    chunk_start = chunk_end + timedelta(days=1)
                    time.sleep(0.5)

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                out      = self.raw_dir / f"firms_{sensor.lower()}_{start_date}_{end_date}.csv"
                combined.to_csv(out, index=False)
                self.logger.info(f"Saved {len(combined)} rows → {out.name}")
            else:
                self.logger.warning(f"No data returned for {sensor}")

    def curate(self):
        """
        - Clean, validate, and filter FIRMS detections. VIIRS only.
        - Handles: sensor filter, confidence, zero-FRP, gas flares,
        Saharan low-FRP, dedup, spatial join to communes.
        """
        import glob

        self.logger.info("Loading raw FIRMS CSV files...")

        csv_files = glob.glob(str(self.raw_dir / "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {self.raw_dir}")

        dfs = []
        for csv in csv_files:
            df_part = pd.read_csv(csv)
            dfs.append(df_part)
            self.logger.info(f"Loaded {csv}: {len(df_part)} rows")

        df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Total rows loaded: {len(df)}")

        # STANDARDIZE 
        df.columns = df.columns.str.lower().str.strip()
        df["acq_date"] = pd.to_datetime(df["acq_date"])
        if "acq_time" in df.columns:
            df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)

        # VIIRS ONLY
        n_before = len(df)
        if "instrument" in df.columns:
            viirs_mask = (
                df["instrument"].str.upper().str.contains("VIIRS", na=False) |
                df["instrument"].str.upper().isin(
                    ["SNPP", "NOAA-20", "NOAA-21", "N20", "N21"]
                )
            )
        elif "satellite" in df.columns:
            viirs_mask = df["satellite"].str.upper().isin(
                ["N", "1", "SNPP", "N20", "N21", "NOAA-20", "NOAA-21"]
            )
        else:
            viirs_mask = pd.Series(True, index=df.index)

        df = df[viirs_mask].copy()
        self.logger.info(f"VIIRS filter: {n_before} → {len(df)}")

        # CONFIDENCE FILTER (keep nominal + high only) 
        n_before = len(df)
        df = df[df["confidence"].astype(str).str.lower().isin(["n", "h"])].copy()
        df["confidence"] = df["confidence"].astype(str).str.lower().map(
            {"n": 66, "h": 99}
        )
        self.logger.info(f"Confidence filter (n/h): {n_before} → {len(df)}")

        # TYPE FILTER — keep only presumed vegetation fires
        n_before = len(df)
        if "type" in df.columns:
            df["type"] = pd.to_numeric(df["type"], errors="coerce")
            df = df[df["type"] == 0].copy()
            self.logger.info(
                f"Type filter (vegetation only): {n_before} → {len(df)} "
                f"(removed {n_before - len(df)} non-vegetation detections)"
            )
        else:
            self.logger.warning("No 'type' column found — skipping type filter")

        # DATA TYPE CONVERSIONS 
        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["frp"]       = pd.to_numeric(df["frp"],       errors="coerce")
        df = df.dropna(subset=["latitude", "longitude", "frp"])

        # ZERO-FRP FILTER 
        # VIIRS cannot physically produce 0 FRP — these are corrupted rows
        n_before = len(df)
        df = df[df["frp"] > 0].copy()
        self.logger.info(
            f"Zero-FRP filter: {n_before} → {len(df)} "
            f"(removed {n_before - len(df)} corrupted rows)"
        )

        # SPATIAL BOUNDS 
        n_before = len(df)
        df = df[
            (df["latitude"]  >= 18.9) & (df["latitude"]  <= 37.1) &
            (df["longitude"] >= -8.7) & (df["longitude"] <= 12.0)
        ].copy()
        self.logger.info(f"Bbox filter: {n_before} → {len(df)}")

        # GAS FLARE FILTER (vectorized haversine) 
        GAS_FLARE_ZONES = [
            {"name": "Hassi Messaoud",      "lat": 31.7, "lon": 6.1, "radius_km": 50},
            {"name": "Hassi R'Mel",         "lat": 32.9, "lon": 3.3, "radius_km": 25},
            {"name": "In Amenas",           "lat": 28.0, "lon": 9.5, "radius_km": 20},
            {"name": "In Salah",            "lat": 27.2, "lon": 2.5, "radius_km": 20},
            {"name": "Ourhoud",             "lat": 28.8, "lon": 6.9, "radius_km": 20},
            {"name": "Rhourde Nouss",       "lat": 29.0, "lon": 7.9, "radius_km": 20},
            {"name": "Illizi",              "lat": 26.5, "lon": 8.5, "radius_km": 25},
            {"name": "Hassi Messaoud Nord", "lat": 32.0, "lon": 6.2, "radius_km": 30},
            {"name": "Haoud Berkaoui",      "lat": 31.4, "lon": 5.9, "radius_km": 25},
        ]

        def build_flare_mask(df, zones):
            mask = pd.Series(False, index=df.index)
            for zone in zones:
                dlat    = np.radians(df["latitude"]  - zone["lat"])
                dlon    = np.radians(df["longitude"] - zone["lon"])
                a       = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(np.radians(zone["lat"]))
                    * np.cos(np.radians(df["latitude"]))
                    * np.sin(dlon / 2) ** 2
                )
                dist_km = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                mask   |= dist_km < zone["radius_km"]
            return mask

        n_before = len(df)
        df = df[~build_flare_mask(df, GAS_FLARE_ZONES)].copy()
        self.logger.info(
            f"Gas flare filter: {n_before} → {len(df)} "
            f"(removed {n_before - len(df)})"
        )

        # SAHARAN LOW-FRP FILTER 
        n_before = len(df)
        df = df[~((df["latitude"] < 30.0) & (df["frp"] < 50))].copy()
        self.logger.info(
            f"Saharan low-FRP filter: {n_before} → {len(df)} "
            f"(removed {n_before - len(df)})"
        )

        # DEDUP (safeguard against overlapping ingest runs) 
        # Same detection can appear in multiple CSV files if date ranges overlap
        n_before = len(df)
        df = df.drop_duplicates(
            subset=["acq_date", "latitude", "longitude"], keep="last"
        )
        if len(df) < n_before:
            self.logger.warning(
                f"Dropped {n_before - len(df)} duplicate detections "
                f"(overlapping ingest date ranges)"
            )

        # SPATIAL JOIN TO COMMUNES 
        self.logger.info("Joining fire detections to communes...")
        communes = gpd.read_file(
            Path(self.config["gadm"]["paths"]["curated"]) / "algeria_communes.gpkg"
        )
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        )
        gdf = gpd.sjoin(
            gdf,
            communes[["GID_1", "GID_2", "NAME_1", "NAME_2", "geometry"]],
            how="left",
            predicate="within"
        )

        n_before = len(gdf)
        gdf = gdf.dropna(subset=["GID_2"]).copy()
        gdf = gdf.drop(columns=["geometry", "index_right"])
        self.logger.info(
            f"Commune join: {n_before} → {len(gdf)} "
            f"(dropped {n_before - len(gdf)} unmatched border points)"
        )

        # FINAL COLUMN SELECTION 
        keep = [
            "acq_date", "acq_time", "latitude", "longitude",
            "frp", "confidence", "daynight",
            "GID_1", "GID_2", "NAME_1", "NAME_2"
        ]
        keep = [c for c in keep if c in gdf.columns]
        gdf  = gdf[keep]

        # SAVE 
        out = self.cur_dir / "firms_curated.parquet"
        self.cur_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(out, index=False)

        self.logger.info(f"Saved {len(gdf)} VIIRS detections → {out}")
        self.logger.info(
            f"Date range: {gdf['acq_date'].min().date()} → "
            f"{gdf['acq_date'].max().date()}"
        )
        self.logger.info(f"Unique communes with fires: {gdf['GID_2'].nunique()}")
        self.logger.info(
            f"FRP stats: min={gdf['frp'].min():.1f}  "
            f"max={gdf['frp'].max():.1f}  "
            f"mean={gdf['frp'].mean():.1f}"
        )


    def load(self):
        out = self.cur_dir / "firms_curated.parquet"
        if not out.exists():
            raise FileNotFoundError("Run curate() first")
        df = pd.read_parquet(out)
        self.logger.info(f"Loaded {len(df)} VIIRS detections from firms_curated.parquet")
        return df


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["firms"]["paths"]
    source = FIRMSSource(config)
    source.ingest("2015-06-01", "2020-10-31")

    source.curate()

    df = source.load()
    print(f"\nFIRMS complete")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['acq_date'].min()} → {df['acq_date'].max()}")
    print(f"   Columns: {list(df.columns)}")
    print(df)