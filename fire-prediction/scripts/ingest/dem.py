# scripts/ingest/dem.py
"""
DEM — Digital Elevation Model (Static)

Ingestion:  confirms DEM file exists
Curation:   clips to Algeria boundary
            reprojects to EPSG:4326
            resamples to canonical 1km grid (from algeria.grid_resolution_m)
            computes slope and aspect
            saves 3 GeoTIFFs:
              curated/dem/elevation.tif
              curated/dem/slope.tif
              curated/dem/aspect.tif
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource
from utils.grid import get_algeria_grid, AlgeriaGrid


def save_raster(array: np.ndarray, path: Path, grid: AlgeriaGrid):
    """Save a numpy array as GeoTIFF aligned to the canonical Algeria grid."""
    with rasterio.open(path, "w", **{
        "driver":    "GTiff",
        "dtype":     "float32",
        "crs":       grid.crs,
        "transform": grid.transform,
        "width":     grid.width,
        "height":    grid.height,
        "count":     1,
        "nodata":    float("nan"),
        "compress":  "lzw",
    }) as dst:
        dst.write(array.astype(np.float32), 1)


def compute_slope_aspect(elevation: np.ndarray, resolution_m: float):
    """Compute slope (degrees) and aspect (degrees) from elevation array."""
    dy, dx     = np.gradient(elevation, resolution_m)
    slope_deg  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect_deg = np.degrees(np.arctan2(-dx, dy)) % 360
    return slope_deg, aspect_deg


class DEMSource(DataSource):

    def ingest(self, start_date=None, end_date=None):
        """Static — confirm GEE-exported DEM file exists."""
        dem_path = self.raw_dir / self.config["dem"]["filename"]
        if not dem_path.exists():
            raise FileNotFoundError(
                f"DEM not found: {dem_path}\n"
                f"Export from GEE then download to data/raw/dem/"
            )
        with rasterio.open(dem_path) as src:
            self.logger.info(f"DEM: {dem_path.name}")
            self.logger.info(f"  CRS: {src.crs} | Shape: {src.shape} | Res: {src.res}")

    def curate(self):
        dem_path      = self.raw_dir / self.config["dem"]["filename"]
        out_dir       = Path(self.config["dem"]["paths"]["curated"])
        boundary_path = Path(self.config["gadm"]["paths"]["curated"]) / "algeria_country.gpkg"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Canonical grid — uses algeria.grid_resolution_m ──────────────────
        res_m = self.config["algeria"]["grid_resolution_m"]
        grid  = get_algeria_grid(boundary_path, res_m)
        self.logger.info(
            f"Grid: {grid.width}×{grid.height} cells at {grid.res_m}m"
        )

        # ── Clip to Algeria ───────────────────────────────────────────────────
        algeria = gpd.read_file(boundary_path).to_crs("EPSG:4326")

        with rasterio.open(dem_path) as src:
            src_crs     = src.crs   # store before closing
            algeria_src = algeria.to_crs(src_crs)
            shapes      = [g.__geo_interface__ for g in algeria_src.geometry]
            clipped, clip_transform = mask(src, shapes, crop=True, nodata=-9999)
            clipped     = clipped[0].astype(np.float32)
            clipped[clipped <= -9999] = np.nan

        self.logger.info(f"Clipped — valid pixels: {np.sum(~np.isnan(clipped))}")

        # ── Reproject and align to canonical grid ─────────────────────────────
        elevation = grid.empty_array()
        reproject(
            source=clipped,
            destination=elevation,
            src_transform=clip_transform,
            src_crs=src_crs,
            dst_transform=grid.transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        self.logger.info(
            f"Elevation: {np.nanmin(elevation):.0f}m → {np.nanmax(elevation):.0f}m"
        )

        # ── Compute slope and aspect ──────────────────────────────────────────
        elev_filled           = np.where(np.isnan(elevation), 0, elevation)
        slope_deg, aspect_deg = compute_slope_aspect(elev_filled, grid.res_m)
        nan_mask              = np.isnan(elevation)
        slope_deg[nan_mask]   = np.nan
        aspect_deg[nan_mask]  = np.nan
        self.logger.info(
            f"Slope: {np.nanmin(slope_deg):.1f}° → {np.nanmax(slope_deg):.1f}°"
        )

        # ── Save ──────────────────────────────────────────────────────────────
        for name, array in [
            ("elevation", elevation),
            ("slope",     slope_deg),
            ("aspect",    aspect_deg),
        ]:
            out_path = out_dir / f"{name}.tif"
            save_raster(array, out_path, grid)
            self.logger.info(f"Saved → {out_path}")

    def load(self):
        out_dir = Path(self.config["dem"]["paths"]["curated"])
        result  = {}
        for name in ["elevation", "slope", "aspect"]:
            path = out_dir / f"{name}.tif"
            if not path.exists():
                raise FileNotFoundError(f"Run curate() first — missing {path}")
            with rasterio.open(path) as src:
                result[name] = src.read(1)
        self.logger.info("Loaded elevation, slope, aspect")
        return result


if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["dem"]["paths"]
    source = DEMSource(config)
    arrays = source.run()

    print(f"\nDEM complete")
    for name, arr in arrays.items():
        print(f"   {name}: shape={arr.shape}, valid={np.sum(~np.isnan(arr))}")