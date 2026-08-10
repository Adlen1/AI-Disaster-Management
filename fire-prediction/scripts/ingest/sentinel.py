"""
Vegetation Indices (Sentinel 2) — Dynamic Data Source

Output files:
    curated/snetinel/sentinel_*.parquet
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource, filter_fire_season


class SentinelSource(DataSource):

    COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
    MAX_CLOUD = 20
    MAX_CLOUD_RETRY = 40
    SCL_MASK_CLASSES = [3, 8, 9, 10]  # cloud shadow, med cloud, high cloud, cirrus
    COMMUNE_BATCH_SIZE = 400   # communes per getInfo() call — keeps each request comfortably sized

    # GEE INIT 

    def _init_gee(self):
        """Requires a Google Cloud project with the Earth Engine API enabled."""
        project = self.config["sentinel"].get("gee_project", "")
        if not project:
            raise RuntimeError(
                "config.yaml -> sentinel.gee_project is empty.\n"
                "Create/enable a GCP project for Earth Engine and set it there."
            )
        try:
            ee.Initialize(project=project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project)

    # CLOUD MASKING 

    @staticmethod
    def _mask_clouds(image: ee.Image) -> ee.Image:
        """Per-pixel cloud/shadow/cirrus mask using the SCL band."""
        scl = image.select("SCL")
        mask = scl.remap(
            SentinelSource.SCL_MASK_CLASSES,
            [0] * len(SentinelSource.SCL_MASK_CLASSES),
            1,
        )
        return (
            image
            .updateMask(mask)
            .select(
                ["B4", "B8", "B11", "B12", "SCL"],
                ["B4", "B8", "B11", "B12", "SCL"],
            )
        )

    # COMMUNE REGIONS FOR reduceRegions() 

    def _load_commune_features(self, communes_path: Path) -> list:
        """
        Load the already-curated commune polygons and convert to a list of
        ee.Feature. Returns a plain list (not yet an ee.FeatureCollection)
        so the caller can batch it — at ~1,541 features this is meaningfully
        bigger than the old 48-wilaya version, so batching happens at the
        ingest() call site rather than building one giant FeatureCollection
        and hoping a single getInfo() covers it.
        """
        communes = gpd.read_file(communes_path).to_crs("EPSG:4326")

        commune_id_col = "GID_2" if "GID_2" in communes.columns else "commune_id"
        commune_name_col = "NAME_2" if "NAME_2" in communes.columns else "commune_name"
        wilaya_id_col = "GID_1" if "GID_1" in communes.columns else "wilaya_id"
        wilaya_name_col = "NAME_1" if "NAME_1" in communes.columns else "wilaya_name"

        geojson = communes.__geo_interface__
        features = []
        for i, feat in enumerate(geojson["features"]):
            row = communes.iloc[i]
            props = {
                "commune_id": row.get(commune_id_col),
                "commune_name": row.get(commune_name_col),
                "wilaya_id": row.get(wilaya_id_col),
                "wilaya_name": row.get(wilaya_name_col),
            }
            features.append(ee.Feature(ee.Geometry(feat["geometry"]), props))

        self.logger.info(f"GEE regions: {len(features)} communes")
        return features

    # ── MONTHLY COMPOSITE ────────────────────────────────────────────────────

    def _get_monthly_composite(self, year: int, month: int, region: ee.Geometry):
        """Build a per-pixel-masked monthly median composite with NDVI/NDWI/NBR."""
        start = f"{year}-{month:02d}-01"
        end = (datetime(year, month, 1) + relativedelta(months=1)).strftime("%Y-%m-%d")

        def build(threshold):
            return (
                ee.ImageCollection(self.COLLECTION)
                .filterBounds(region)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", threshold))
                .map(self._mask_clouds)
            )

        col = build(self.MAX_CLOUD)
        count = col.size().getInfo()

        if count == 0:
            self.logger.warning(
                f"  {year}-{month:02d}: no images <{self.MAX_CLOUD}% cloud "
                f"— retrying <{self.MAX_CLOUD_RETRY}%"
            )
            col = build(self.MAX_CLOUD_RETRY)
            count = col.size().getInfo()

        if count == 0:
            return None, start

        composite = col.median()
        ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndwi = composite.normalizedDifference(["B8", "B11"]).rename("NDWI")
        nbr = composite.normalizedDifference(["B8", "B12"]).rename("NBR")

        return composite.addBands([ndvi, ndwi, nbr]).select(["NDVI", "NDWI", "NBR"]), start

    def _reduce_batch(self, composite, batch_features: list, label: str,
                       batch_num: int, n_batches: int) -> list:
        """
        Run reduceRegions() + getInfo() on one batch of commune features,
        with the same retry pattern as before. Returns a list of row dicts.
        """
        batch_fc = ee.FeatureCollection(batch_features)
        reduced = composite.reduceRegions(
            collection=batch_fc,
            reducer=ee.Reducer.mean(),
            scale=20,
            tileScale=8,
        )

        for attempt in range(1, 4):
            try:
                self.logger.info(
                    f"  {label}: batch {batch_num}/{n_batches} "
                    f"({len(batch_features)} communes), attempt {attempt}/3..."
                )
                features = reduced.getInfo()["features"]
                return [feat["properties"] for feat in features]
            except Exception as e:
                self.logger.warning(
                    f"  {label}: batch {batch_num}/{n_batches} attempt {attempt} failed — {e}"
                )
                if attempt < 3:
                    time.sleep(10 * attempt)
                else:
                    self.logger.error(
                        f"  {label}: batch {batch_num}/{n_batches} failed all retries — "
                        f"{len(batch_features)} communes will be MISSING for this month"
                    )
                    return []

    def ingest(self, start_date: str = None, end_date: str = None):
        """
        - For each month: build cloud-masked composite -> reduceRegions() over
        commune batches -> mean NDVI/NDWI/NBR per commune -> getInfo() -> CSV.
        - No GeoTIFF, no manual Drive download, no grid-building step.
        - Each month produces one CSV: raw/sentinel/sentinel2_YYYYMM.csv
        """
        self._init_gee()

        if not start_date:
            start_date = datetime.today().replace(day=1).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.today().strftime("%Y-%m-%d")

        if start_date < "2015-06-01":
            self.logger.warning("Adjusting start to 2015-06-01 (Sentinel-2 availability)")
            start_date = "2015-06-01"

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")

        communes_path = Path(self.config["gadm"]["paths"]["curated"]) / "algeria_communes.gpkg"
        if not communes_path.exists():
            raise FileNotFoundError(
                f"{communes_path} not found — run the GADM ingest/curate step first."
            )

        self.logger.info(f"Sentinel-2 ingestion: {start_date} -> {end_date}")
        self.logger.info("Mode: reduceRegions() over communes (batched) -> table (no GeoTIFF)")

        all_features = self._load_commune_features(communes_path)
        assert 1500 <= len(all_features) <= 1600, \
            f"Unexpected commune count: {len(all_features)}"

        # Use simple bbox for the Sentinel search region 
        bbox = self.config["algeria"]["bbox"]
        region = ee.Geometry.Rectangle(bbox)

        batches = [
            all_features[i:i + self.COMMUNE_BATCH_SIZE]
            for i in range(0, len(all_features), self.COMMUNE_BATCH_SIZE)
        ]
        self.logger.info(
            f"Batching {len(all_features)} communes into {len(batches)} "
            f"batches of up to {self.COMMUNE_BATCH_SIZE}"
        )

        fire_months = self.config.get("training", {}).get(
            "fire_season_months", list(range(1, 13))
        )

        current         = start.replace(day=1)
        success, skipped, failed = 0, 0, []

        while current <= end:
            year, month = current.year, current.month

            if month not in fire_months:
                self.logger.info(f"  {year}{month:02d}: not fire season — skipping")
                current += relativedelta(months=1)
                continue

            label   = f"{year}{month:02d}"
            out_csv = self.raw_dir / f"sentinel2_{label}.csv"

            if out_csv.exists() and out_csv.stat().st_size > 100:
                self.logger.info(f"  {label}: already exists — skipping")
                skipped += 1
                current += relativedelta(months=1)
                continue

            self.logger.info(f"  {label}: building composite...")
            composite, _ = self._get_monthly_composite(year, month, region)

            if composite is None:
                self.logger.warning(f"  {label}: no usable images — skipping")
                failed.append(label)
                current += relativedelta(months=1)
                continue

            all_rows = []
            for batch_num, batch_features in enumerate(batches, start=1):
                props_list = self._reduce_batch(
                    composite, batch_features, label, batch_num, len(batches)
                )
                for props in props_list:
                    all_rows.append({
                        "date"        : f"{year}-{month:02d}-01",  # first-of-month; join on year+month
                        "year"        : year,
                        "month"       : month,
                        "commune_id"  : props.get("commune_id"),
                        "commune_name": props.get("commune_name"),
                        "wilaya_id"   : props.get("wilaya_id"),
                        "wilaya_name" : props.get("wilaya_name"),
                        "NDVI"        : props.get("NDVI"),
                        "NDWI"        : props.get("NDWI"),
                        "NBR"         : props.get("NBR"),
                    })

            if not all_rows:
                self.logger.error(f"  {label}: all batches failed — no data for this month")
                failed.append(label)
                current += relativedelta(months=1)
                continue

            df_month = pd.DataFrame(all_rows)
            before   = len(df_month)

            # Log cloud gap rate but do NOT drop here — curate() will forward-fill
            n_missing = df_month["NDVI"].isna().sum()
            self.logger.info(
                f"  {label}: {before - n_missing}/{before} communes with valid NDVI "
                f"({n_missing} cloud gaps — will be forward-filled in curate())"
            )

            df_month.to_csv(out_csv, index=False)
            self.logger.info(f"  {label}: saved -> {out_csv.name}")
            success += 1

            current += relativedelta(months=1)
            time.sleep(1)

        self.logger.info(
            f"\nIngestion summary: downloaded={success} skipped={skipped} "
            f"failed={len(failed)} {failed if failed else ''}"
        )


    def curate(self):
        """
        - Merge monthly CSVs 
        - forward-fill cloud gaps 
        - apply fire-season filter
        - save parquet.
        """
        out_dir = Path(self.config["sentinel"]["paths"]["curated"])
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(self.raw_dir.glob("sentinel2_*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No Sentinel-2 CSVs in {self.raw_dir}\nRun ingest() first."
            )

        self.logger.info(f"Loading {len(csv_files)} monthly Sentinel-2 CSVs")

        dfs = []
        for f in csv_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                self.logger.warning(f"  Skipping {f.name}: {e}")

        df = pd.concat(dfs, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])

        for col in ["NDVI", "NDWI", "NBR"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(-1, 1)

        self.logger.info(f"Combined shape before gap-fill: {df.shape}")
        self.logger.info(f"Communes represented: {df['commune_id'].nunique()}")

        # FORWARD-FILL cloud gaps per commune (then backfill for season start) 
        # Vegetation state changes slowly — previous month's value is a valid proxy.
        df = df.sort_values(["commune_id", "date"])
        for col in ["NDVI", "NDWI", "NBR"]:
            df[col] = (
                df.groupby("commune_id")[col]
                .transform(lambda x: x.ffill().bfill())
                .round(4)
            )

        # Log how many gaps remain (should be zero unless a commune has NO valid
        # observations across the entire time series — extremely rare)
        remaining_nan = df["NDVI"].isna().sum()
        if remaining_nan > 0:
            bad_communes = df[df["NDVI"].isna()]["commune_id"].unique()
            self.logger.warning(
                f"{remaining_nan} rows still NaN after gap-fill "
                f"({len(bad_communes)} communes with zero valid observations) — "
                f"these will be dropped: {bad_communes[:10]}"
            )
            df = df.dropna(subset=["NDVI"])

        # FIRE SEASON FILTER
        df = filter_fire_season(df, self.config, date_col="date")
        self.logger.info(f"Final shape after fire-season filter: {df.shape}")

        if df.empty:
            raise ValueError(
                "No rows remain after the fire-season filter.\n"
                "Make sure you ingested fire-season months (Jun-Sep) and re-run curate()."
            )

        self.logger.info(f"Date range: {df['date'].min()} -> {df['date'].max()}")
        self.logger.info(f"NDVI stats: mean={df['NDVI'].mean():.3f} "
                        f"min={df['NDVI'].min():.3f} max={df['NDVI'].max():.3f}")

        start_str = df["date"].min().strftime("%Y%m%d")
        end_str   = df["date"].max().strftime("%Y%m%d")
        out_path  = out_dir / f"sentinel_{start_str}_{end_str}.parquet"
        df.to_parquet(out_path, index=False)
        self.logger.info(f"Saved -> {out_path}")

    def load(self):
        out_dir = Path(self.config["sentinel"]["paths"]["curated"])
        files = sorted(out_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError("Run curate() first")
        df = pd.read_parquet(files[-1])
        self.logger.info(f"Loaded Sentinel-2: {df.shape}")
        return df


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["sentinel"]["paths"]
    source = SentinelSource(config)

    source.ingest(start_date="2015-07-01", end_date="2015-07-31")  # fire-season test month
    source.curate()

    df = source.load()
    print(f"\nSentinel-2 complete")
    print(f"   Shape:   {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(df)