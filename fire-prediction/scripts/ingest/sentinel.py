# scripts/ingest/sentinel.py
"""
Sentinel-2 — Vegetation Indices (Dynamic)

Architecture: GEE-side computation, table output, no raster download.
  1. Build monthly median composite (per-pixel cloud-masked)
  2. Compute NDVI, NDWI (Gao/NDMI), NBR inside GEE
  3. reduceRegions() over the 48 wilaya polygons (NOT a fine pixel grid —
     see note below) -> mean index value per wilaya per month
  4. getInfo() -> small table (48 rows/month) -> CSV -> parquet

Why wilaya-level, not a fine 1km grid:
  Algeria's bbox at 1km resolution is ~4.6 million cells (see your DEM log:
  shape 2013x2293). Running reduceRegions() over millions of regions and
  pulling results back with getInfo() is not viable — GEE's reduceToVectors
  is meant to vectorize contiguous same-valued raster regions (e.g. land
  cover classes), not tile a country into a uniform grid, and getInfo() has
  a practical payload limit far below millions of rows regardless. This is
  the same row-explosion problem already solved for ERA5 (kept at its
  native ~31km grid instead of the fine terrain grid) — recreated here one
  layer up in the grid-building step. Wilaya polygons (48 features) keep
  this fast, reliable, and consistent with how FIRMS historical labels are
  already joined.

Indices:
  NDVI = (B8 - B4)  / (B8 + B4)    vegetation greenness
  NDWI = (B8 - B11) / (B8 + B11)   canopy moisture (Gao's NDMI, not McFeeters')
  NBR  = (B8 - B12) / (B8 + B12)   fuel dryness / burn ratio proxy

Two modes — same function:
  Training:    ingest("2015-06-01", "2025-10-31")
  Operational: ingest("2026-07-09", "2026-07-09") -> latest available
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

    # ── GEE INIT ─────────────────────────────────────────────────────────────

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

    # ── CLOUD MASKING ────────────────────────────────────────────────────────

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

    # ── WILAYA REGIONS FOR reduceRegions() ──────────────────────────────────

    def _build_gee_wilaya_regions(self, wilaya_path: Path) -> ee.FeatureCollection:
        """
        Load the already-curated wilaya polygons and convert to an
        ee.FeatureCollection. 48 features — trivially small for reduceRegions
        + getInfo(), and reuses the boundary file you already built instead
        of generating a new grid.
        """
        wilayas = gpd.read_file(wilaya_path).to_crs("EPSG:4326")

        id_col = "GID_1" if "GID_1" in wilayas.columns else wilayas.columns[0]
        name_col = "NAME_1" if "NAME_1" in wilayas.columns else None

        geojson = wilayas.__geo_interface__
        features = []
        for i, feat in enumerate(geojson["features"]):
            props = {"wilaya_id": wilayas.iloc[i][id_col]}
            if name_col:
                props["wilaya_name"] = wilayas.iloc[i][name_col]
            features.append(ee.Feature(ee.Geometry(feat["geometry"]), props))

        self.logger.info(f"GEE regions: {len(features)} wilayas")
        return ee.FeatureCollection(features)

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

    # ── INGESTION — table-only, no raster download ──────────────────────────

    def ingest(self, start_date: str = None, end_date: str = None):
        """
        For each month: build cloud-masked composite -> reduceRegions() over
        the 48 wilayas -> mean NDVI/NDWI/NBR per wilaya -> getInfo() -> CSV.

        No GeoTIFF, no manual Drive download, no grid-building step.
        Each month produces one small CSV: raw/sentinel/sentinel2_YYYYMM.csv
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
        end = datetime.strptime(end_date, "%Y-%m-%d")
        bbox = self.config["algeria"]["bbox"]
        region = ee.Geometry.Rectangle(bbox)

        wilaya_path = Path(self.config["gadm"]["paths"]["curated"]) / "algeria_wilayas.gpkg"
        if not wilaya_path.exists():
            raise FileNotFoundError(
                f"{wilaya_path} not found — run the GADM ingest/curate step first."
            )

        self.logger.info(f"Sentinel-2 ingestion: {start_date} -> {end_date}")
        self.logger.info("Mode: reduceRegions() over wilayas -> table (no GeoTIFF)")

        gee_regions = self._build_gee_wilaya_regions(wilaya_path)

        current = start.replace(day=1)
        success, skipped, failed = 0, 0, []

        fire_months = self.config.get("training", {}).get(
            "fire_season_months", list(range(1, 13))
        )

        while current <= end:

            year, month = current.year, current.month

            if month not in fire_months:
                self.logger.info(f"  {year}{month:02d}: not fire season — skipping")
                current += relativedelta(months=1)
                continue

            label = f"{year}{month:02d}"
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

            reduced = composite.reduceRegions(
                collection=gee_regions,
                reducer=ee.Reducer.mean(),
                scale=1000,
                tileScale=4,   # splits computation into smaller tiles to
                               # avoid "User memory limit exceeded" — GEE's
                               # standard fix for this exact error. Increase
                               # to 8 or 16 if it still fails.
            )

            for attempt in range(1, 4):
                try:
                    self.logger.info(f"  {label}: fetching table (attempt {attempt}/3)...")
                    features = reduced.getInfo()["features"]  # 48 features — fast, safe size
                    rows = []
                    for feat in features:
                        props = feat["properties"]
                        rows.append({
                            "date": f"{year}-{month:02d}-01",
                            "year": year,
                            "month": month,
                            "wilaya_id": props.get("wilaya_id"),
                            "wilaya_name": props.get("wilaya_name"),
                            "NDVI": props.get("NDVI"),
                            "NDWI": props.get("NDWI"),
                            "NBR": props.get("NBR"),
                        })

                    df_month = pd.DataFrame(rows)
                    before = len(df_month)
                    df_month = df_month.dropna(subset=["NDVI"]).copy()
                    self.logger.info(
                        f"  {label}: {len(df_month)}/{before} wilayas with valid NDVI"
                    )

                    df_month.to_csv(out_csv, index=False)
                    self.logger.info(f"  {label}: saved -> {out_csv.name}")
                    success += 1
                    break

                except Exception as e:
                    self.logger.warning(f"  {label}: attempt {attempt} failed — {e}")
                    if attempt < 3:
                        time.sleep(10 * attempt)
                    else:
                        self.logger.error(f"  {label}: all retries failed")
                        failed.append(label)
                        if out_csv.exists():
                            out_csv.unlink()

            current += relativedelta(months=1)
            time.sleep(1)

        self.logger.info(
            f"\nIngestion summary: downloaded={success} skipped={skipped} "
            f"failed={len(failed)} {failed if failed else ''}"
        )

    # ── CURATION ─────────────────────────────────────────────────────────────

    def curate(self):
        """Merge monthly CSVs, apply fire-season filter, save parquet."""
        out_dir = Path(self.config["sentinel"]["paths"]["curated"])
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(self.raw_dir.glob("sentinel2_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No Sentinel-2 CSVs in {self.raw_dir}\nRun ingest() first.")

        self.logger.info(f"Loading {len(csv_files)} monthly Sentinel-2 CSVs")

        dfs = []
        for f in csv_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                self.logger.warning(f"  Skipping {f.name}: {e}")

        df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Combined shape before fire-season filter: {df.shape}")

        df["date"] = pd.to_datetime(df["date"])
        for col in ["NDVI", "NDWI", "NBR"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4).clip(-1, 1)

        df = filter_fire_season(df, self.config, date_col="date")
        self.logger.info(f"Final shape after fire-season filter: {df.shape}")

        if df.empty:
            raise ValueError(
                "No rows remain after the fire-season filter. This means either:\n"
                "  1. The ingested CSVs only covered non-fire-season months "
                "(e.g. only December was downloaded) — ingest fire-season "
                "months (May-Oct) and re-run curate(), or\n"
                "  2. Every fire-season month happened to have all-cloudy "
                "wilayas (unlikely but possible) — check the per-month "
                "'X/48 wilayas with valid NDVI' log lines from ingest() above."
            )

        self.logger.info(f"Date range: {df['date'].min()} -> {df['date'].max()}")
        self.logger.info(f"NDVI range: {df['NDVI'].min():.3f} -> {df['NDVI'].max():.3f}")

        start_str = df["date"].min().strftime("%Y%m%d")
        end_str = df["date"].max().strftime("%Y%m%d")
        out_path = out_dir / f"sentinel_{start_str}_{end_str}.parquet"
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