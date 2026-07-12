"""
NASA FIRMS — Fire Archive + NRT (Dynamic)

For new data: set FIRMS_MAP_KEY in .env then call ingest(start, end)

Output: curated/firms/firms_algeria_YYYYMMDD_YYYYMMDD.parquet
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

    # ── INGESTION ─────────────────────────────────────────────────────────────

    def ingest(self, start_date: str = None, end_date: str = None):
        """
        Download FIRMS data for Algeria via API.
        Skip this if data already in raw/firms/ as CSV.
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

        # Skip if raw files already exist
        existing = list(self.raw_dir.glob("*.csv"))
        if existing:
            self.logger.info(
                f"Found {len(existing)} existing CSV(s) in data/raw/firms/ — "
                f"skipping API ingestion. Delete raw files to force re-download."
            )
            return

        # Fire season filter for API ingestion
        fire_months = self.config.get("training", {}).get(
            "fire_season_months", list(range(1, 13))
        )

        for sensor in sensors:
            all_dfs     = []
            chunk_start = start

            with tqdm(total=total_days, desc=f"{sensor}", unit="days") as pbar:
                while chunk_start <= end:

                    # Jump entire non-fire-season months
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

    # ── CURATION ──────────────────────────────────────────────────────────────

    def curate(self):
        """
        Process all CSV files in raw/firms/ — both archive and NRT.
        Works on already-downloaded files without needing the API.
        """
        out_dir   = Path(self.config["firms"]["paths"]["curated"])
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(self.raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files in {self.raw_dir}\n"
                f"Either run ingest() or copy your downloaded FIRMS CSVs here."
            )

        self.logger.info(f"Loading {len(csv_files)} FIRMS CSV files")

        # ── Load all CSVs ─────────────────────────────────────────────────────
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df["source_file"] = f.name
                dfs.append(df)
                self.logger.debug(f"  {f.name}: {len(df)} rows")
            except Exception as e:
                self.logger.warning(f"  Skipping {f.name}: {e}")

        df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Combined: {len(df)} rows")

        # ── Parse dates ───────────────────────────────────────────────────────
        df["acq_date"] = pd.to_datetime(df["acq_date"])

        # ── Confidence filter — VIIRS and MODIS use different formats ─────────
        viirs_mask = df["instrument"].str.upper().str.contains("VIIRS", na=False)
        modis_mask = df["instrument"].str.upper().str.contains("MODIS", na=False)

        viirs = df[viirs_mask].copy()
        modis = df[modis_mask].copy()
        other = df[~viirs_mask & ~modis_mask].copy()

        # VIIRS: string labels
        n = len(viirs)
        viirs = viirs[viirs["confidence"].astype(str).isin(["n", "h"])]
        self.logger.info(f"VIIRS confidence: {n} → {len(viirs)} (dropped {n - len(viirs)})")

        # MODIS: integer 0-100
        n = len(modis)
        modis["confidence"] = pd.to_numeric(modis["confidence"], errors="coerce")
        modis = modis[modis["confidence"] >= 50]
        self.logger.info(f"MODIS confidence: {n} → {len(modis)} (dropped {n - len(modis)})")

        df = pd.concat([viirs, modis, other], ignore_index=True)
        df["confidence"] = df["confidence"].astype(str)

        # ── Vegetation fires only ─────────────────────────────────────────────
        if "type" in df.columns:
            n  = len(df)
            df = df[df["type"].isna() | (df["type"] == 0)]
            self.logger.info(f"Type filter: {n} → {len(df)} (dropped {n - len(df)})")

        # ── Deduplicate ───────────────────────────────────────────────────────
        n  = len(df)
        df = df.drop_duplicates(
            subset=["latitude", "longitude", "acq_date", "acq_time", "instrument"]
        )
        self.logger.info(f"Deduplication: {n} → {len(df)} (dropped {n - len(df)})")

        # ── Add derived columns ───────────────────────────────────────────────
        df["year"]       = df["acq_date"].dt.year
        df["month"]      = df["acq_date"].dt.month
        df["day_of_year"]= df["acq_date"].dt.dayofyear
        df               = df.sort_values("acq_date").reset_index(drop=True)

        # ── EDA summary ───────────────────────────────────────────────────────
        self.logger.info(f"Final: {df.shape}")
        self.logger.info(f"Date range: {df['acq_date'].min()} → {df['acq_date'].max()}")

        by_year = df.groupby("year").size()
        self.logger.info(f"Detections per year:\n{by_year.to_string()}")

        fire_season_pct = 100 * df["month"].isin([7, 8, 9]).sum() / len(df)
        self.logger.info(
            f"Fire season Jul-Sep: {fire_season_pct:.1f}% "
            f"({'Good' if fire_season_pct > 50 else 'check'})"
        )

        # Drop metadata columns not needed for training
        df = df.drop(columns=["version"], errors="ignore")

        df = filter_fire_season(df, self.config, date_col="acq_date")

        boundary_path = Path(self.config["gadm"]["paths"]["curated"]) / "algeria_wilayas.gpkg"
        
        
        wilayas = (
            gpd.read_file(boundary_path)[["GID_1", "NAME_1", "geometry"]]
            .rename(columns={
                "GID_1": "wilaya_id",
                "NAME_1": "wilaya_name"
            })
        )

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        gdf = gpd.sjoin(gdf, wilayas, how="left", predicate="within").drop(columns=["geometry", "index_right"])
        df  = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

        # ── Save as parquet ───────────────────────────────────────────────────
        start_str = df["acq_date"].min().strftime("%Y%m%d")
        end_str   = df["acq_date"].max().strftime("%Y%m%d")
        out_path  = out_dir / f"firms_algeria_{start_str}_{end_str}.parquet"
        df.to_parquet(out_path, index=False)
        self.logger.info(f"Saved → {out_path}")

    # ── LOAD ──────────────────────────────────────────────────────────────────

    def load(self):
        """Return most recent curated FIRMS parquet."""
        out_dir = Path(self.config["firms"]["paths"]["curated"])
        files   = sorted(out_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError("Run curate() first")
        df = pd.read_parquet(files[-1])
        self.logger.info(f"Loaded {len(df)} fire detections from {files[-1].name}")
        return df


if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["firms"]["paths"]
    source = FIRMSSource(config)
    source.ingest( "2015-06-01", "2025-10-31")

    source.curate()

    df = source.load()
    print(f"\nFIRMS complete")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['acq_date'].min()} → {df['acq_date'].max()}")
    print(f"   Columns: {list(df.columns)}")
    print(df)