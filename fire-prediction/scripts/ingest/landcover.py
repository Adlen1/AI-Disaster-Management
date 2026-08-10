# scripts/ingest/landcover.py
"""
ESA WorldCover 2021 — Static Data Source

Output files:
    raw/landcover/worldcover_*.tif
    curated/landcover/landcover_communes.parquet
"""
import sys
from collections import defaultdict

import requests
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from rasterstats import zonal_stats

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource


class LandCoverSource(DataSource):

    WORLDCOVER_URL = (
        "https://esa-worldcover.s3.amazonaws.com/v200/2021/map/"
        "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    )

    ALGERIA_TILES = [
        # Row N18 (lat 18-21N) 
        "N18W009", "N18W006", "N18W003", "N18E000", "N18E003",
        "N18E006", "N18E009", "N18E012",
        # Row N21 (lat 21-24N)
        "N21W009", "N21W006", "N21W003", "N21E000", "N21E003",
        "N21E006", "N21E009", "N21E012",
        # Row N24 (lat 24-27N)
        "N24W009", "N24W006", "N24W003", "N24E000", "N24E003",
        "N24E006", "N24E009", "N24E012",
        # Row N27 (lat 27-30N)
        "N27W009", "N27W006", "N27W003", "N27E000", "N27E003",
        "N27E006", "N27E009", "N27E012",
        # Row N30 (lat 30-33N)
        "N30W009", "N30W006", "N30W003", "N30E000", "N30E003",
        "N30E006", "N30E009", "N30E012",
        # Row N33 (lat 33-36N)
        "N33W009", "N33W006", "N33W003", "N33E000", "N33E003",
        "N33E006", "N33E009", "N33E012",
        # Row N36 (lat 36-39N)
        "N36W009", "N36W006", "N36W003", "N36E000", "N36E003",
        "N36E006", "N36E009", "N36E012",
    ]

    FUEL_CLASSES = {
        "forest_fraction":   [10],
        "shrub_fraction":    [20],
        "grass_fraction":    [30],
        "crop_fraction":     [40],
        "urban_fraction":    [50],
        "bare_fraction":     [60],
        "burnable_fraction": [10, 20, 30],
    }

    def ingest(self, start_date: str = None, end_date: str = None) -> None:
        """
        - Downloads ESA WorldCover tiles for Algeria
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        skipped    = 0

        for tile in self.ALGERIA_TILES:
            url = self.WORLDCOVER_URL.format(tile=tile)
            out = self.raw_dir / f"worldcover_{tile}.tif"

            if out.exists():
                self.logger.info(f"Already exists — skipping {tile}")
                skipped += 1
                continue

            self.logger.info(f"Downloading {tile}...")
            try:
                r = requests.get(url, stream=True, timeout=120)
                if r.status_code == 200:
                    with open(out, "wb") as f:
                        for chunk in r.iter_content(1024 * 1024):
                            f.write(chunk)
                    self.logger.info(f"Saved → {out.name}")
                    downloaded += 1
                else:
                    self.logger.warning(f"Tile {tile} not found (HTTP {r.status_code}) — skipping")
            except Exception as e:
                self.logger.error(f"Failed to download {tile}: {e}")

        self.logger.info(f"Ingest complete: {downloaded} downloaded, {skipped} skipped")

    def _load_polygons(self):
        """
        - Load commune polygons and normalize GADM's raw column names.
        """
        boundaries_dir = Path(self.config.get("gadm", {}).get(
            "paths", {}).get("curated", "data/curated/boundaries"))

        communes_path = boundaries_dir / "algeria_communes.gpkg"
        if communes_path.exists():
            polygons = gpd.read_file(communes_path).to_crs("EPSG:4326")
            polygons = polygons.rename(columns={
                "GID_2": "commune_id", "NAME_2": "commune_name",
                "GID_1": "wilaya_id", "NAME_1": "wilaya_name",
            })
            id_col, name_col = "commune_id", "commune_name"
        else:
            wilayas_path = boundaries_dir / "algeria_wilayas.gpkg"
            polygons = gpd.read_file(wilayas_path).to_crs("EPSG:4326")
            polygons = polygons.rename(columns={
                "GID_1": "wilaya_id", "NAME_1": "wilaya_name",
            })
            id_col, name_col = "wilaya_id", "wilaya_name"
            self.logger.warning("Communes not found — falling back to wilayas")

        missing = [c for c in (id_col, name_col) if c not in polygons.columns]
        if missing:
            raise ValueError(
                f"Expected columns {missing} not found after renaming — "
                f"check the actual column names in the boundary file "
                f"(GADM version/layer may differ from what's assumed here)."
            )

        return polygons, id_col, name_col

    def curate(self) -> None:
        """
        - Compute land cover fractions per commune polygon.
        - for each tile, read only the windowed extent
        of each polygon (via rasterio) rather than loading the full tile.
        RAM stays bounded to one small polygon window at a time.
        """
        import rasterio
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import mapping

        polygons, id_col, name_col = self._load_polygons()
        n_polygons = len(polygons)

        tifs = sorted(self.raw_dir.glob("worldcover_*.tif"))
        if not tifs:
            raise FileNotFoundError(f"No WorldCover tiles in {self.raw_dir}")

        self.logger.info(
            f"Computing land cover for {n_polygons} polygons "
            f"across {len(tifs)} tiles (windowed reads — RAM safe)..."
        )

        accumulated = [defaultdict(int) for _ in range(n_polygons)]

        for tif in tifs:
            self.logger.info(f"  Processing {tif.name}...")

            try:
                with rasterio.open(tif) as src:
                    tile_bounds = src.bounds

                    for i, row in enumerate(polygons.itertuples()):
                        geom = row.geometry

                        # Quick bbox check — skip polygons fully outside this tile
                        b = geom.bounds  # (minx, miny, maxx, maxy)
                        if (b[2] < tile_bounds.left  or
                            b[0] > tile_bounds.right or
                            b[3] < tile_bounds.bottom or
                            b[1] > tile_bounds.top):
                            continue

                        try:
                            # Read only the pixels within this polygon's window
                            out_image, _ = rio_mask(
                                src,
                                [mapping(geom)],
                                crop=True,
                                nodata=0,
                                all_touched=True,
                            )
                            pixels = out_image[0]  # shape: (H, W)
                            pixels = pixels[pixels != 0]  # exclude nodata

                            if pixels.size == 0:
                                continue

                            # Count pixels per class
                            classes, counts = np.unique(pixels, return_counts=True)
                            for cls, cnt in zip(classes, counts):
                                accumulated[i][int(cls)] += int(cnt)

                        except Exception as e:
                            # Polygon may clip to empty (border commune touching tile edge)
                            self.logger.debug(f"    Polygon {i} skip on {tif.name}: {e}")
                            continue

            except Exception as e:
                self.logger.warning(f"  Could not open {tif.name}: {e}")
                continue

            self.logger.info(f"  Done: {tif.name}")

        # Build result dataframe
        result = polygons[[id_col]].copy()
        if name_col in polygons.columns:
            result[name_col] = polygons[name_col].values
        if "wilaya_id" in polygons.columns and id_col != "wilaya_id":
            result["wilaya_id"]   = polygons["wilaya_id"].values
        if "wilaya_name" in polygons.columns and id_col != "wilaya_id":
            result["wilaya_name"] = polygons["wilaya_name"].values

        for col, classes in self.FUEL_CLASSES.items():
            fracs = []
            for s in accumulated:
                total = sum(s.values())
                count = sum(s.get(c, 0) for c in classes)
                fracs.append(round(count / total, 4) if total > 0 else 0.0)
            result[col] = fracs
            self.logger.info(
                f"  {col}: mean={np.mean(fracs):.3f}  max={np.max(fracs):.3f}"
            )

        # Sanity check — communes with zero pixels across all tiles
        zero_mask = [sum(s.values()) == 0 for s in accumulated]
        n_zero = sum(zero_mask)
        if n_zero > 0:
            zero_ids = polygons[id_col].values[zero_mask]
            self.logger.warning(
                f"{n_zero} communes had zero land cover pixels "
                f"(likely tiny border communes) — fractions will be 0.0: "
                f"{zero_ids[:10]}"
            )

        out_path = self.cur_dir / "landcover_communes.parquet"
        self.cur_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False)
        self.logger.info(f"Saved -> {out_path}  ({len(result)} rows)")


    def load(self) -> pd.DataFrame:
        """Return curated land cover fractions."""
        files = sorted(self.cur_dir.glob("landcover_*.parquet"))
        if not files:
            raise FileNotFoundError("Run curate() first")
        df = pd.read_parquet(files[-1])
        self.logger.info(f"Loaded land cover: {df.shape}")
        return df

# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger

    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["landcover"]["paths"]

    source = LandCoverSource(config)

    source.ingest()
    source.curate()

    df = source.load()

    print("\nLandCover complete")
    print(f"   Shape:        {df.shape}")

    if "commune_id" in df.columns:
        print(f"   Communes:     {df['commune_id'].nunique()}")
    elif "wilaya_id" in df.columns:
        print(f"   Wilayas:      {df['wilaya_id'].nunique()}")

    print(f"   Columns:      {list(df.columns)}")

    fraction_cols = [
        "forest_fraction", "shrub_fraction", "grass_fraction", "crop_fraction",
        "urban_fraction", "bare_fraction", "burnable_fraction",
    ]

    print("\nMean fractions:")
    for col in fraction_cols:
        if col in df.columns:
            print(f"   {col:<20}: {df[col].mean():.3f}")

    print("\nFirst 5 rows:")
    print(df.head())