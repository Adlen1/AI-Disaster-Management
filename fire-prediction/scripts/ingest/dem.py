"""
DEM — Digital Elevation Model (Static)

Ingestion:  confirms DEM file exists
Curation:   clips to Algeria boundary
            reprojects to EPSG:4326
            resamples to 1km resolution
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
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource
from utils.grid import get_algeria_grid


def compute_slope_aspect(elevation: np.ndarray, resolution_m: float):
    """
    Compute slope (degrees) and aspect (degrees) from elevation array.
    Uses numpy gradient — matches how GEE and GDAL compute these.
    
    Args:
        elevation:    2D numpy array of elevation values (meters)
        resolution_m: pixel size in meters
    Returns:
        slope_deg, aspect_deg — same shape as elevation
    """
    # Gradient in x (east) and y (north) directions
    dy, dx = np.gradient(elevation, resolution_m)

    # Slope in degrees
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect in degrees (0=North, 90=East, 180=South, 270=West)
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad) % 360

    return slope_deg, aspect_deg


class DEMSource(DataSource):

    def ingest(self, start_date=None, end_date=None):
        """Static — confirm file exists and is a valid raster."""
        dem_path = self.raw_dir / self.config["dem"]["filename"]

        if not dem_path.exists():
            raise FileNotFoundError(
                f"DEM file not found: {dem_path}\n"
                f"Download SRTM DEM for Algeria from:\n"
                f"  https://code.earthengine.google.com/"
            )

        with rasterio.open(dem_path) as src:
            self.logger.info(f"DEM file: {dem_path.name}")
            self.logger.info(f"  CRS:        {src.crs}")
            self.logger.info(f"  Resolution: {src.res}")
            self.logger.info(f"  Shape:      {src.shape}")
            self.logger.info(f"  Bounds:     {src.bounds}")
            self.logger.info(f"  NoData:     {src.nodata}")


    def curate(self):
        """
        Full curation pipeline:
        1. Load Algeria boundary for clipping
        2. Clip DEM to Algeria
        3. Reproject to EPSG:4326 if needed
        4. Resample to target resolution
        5. Compute slope and aspect
        6. Save all three bands as separate GeoTIFFs
        """
        dem_path     = self.raw_dir  / self.config["dem"]["filename"]
        out_dir      = Path(self.config["dem"]["paths"]["curated"])
        boundary_path = Path(self.config["gadm"]["paths"]["curated"]) / "algeria_country.gpkg"
        target_res_m = self.config["dem"]["output_resolution_m"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Load Algeria boundary ──────────────────────────────────────────
        self.logger.info("Loading Algeria boundary for clipping")
        algeria = gpd.read_file(boundary_path).to_crs("EPSG:4326")
        shapes  = [geom.__geo_interface__ for geom in algeria.geometry]

        # ── 2. Clip and reproject ─────────────────────────────────────────────
        self.logger.info("Clipping DEM to Algeria boundary")
        with rasterio.open(dem_path) as src:
            # Reproject boundary to match DEM CRS for clipping
            algeria_dem_crs = algeria.to_crs(src.crs)
            clip_shapes = [g.__geo_interface__ for g in algeria_dem_crs.geometry]

            clipped, clip_transform = mask(src, clip_shapes, crop=True, nodata=-9999)
            clipped = clipped[0].astype(np.float32)
            clipped[clipped == -9999] = np.nan

            clip_profile = src.profile.copy()
            clip_profile.update({
                "height":    clipped.shape[0],
                "width":     clipped.shape[1],
                "transform": clip_transform,
                "dtype":     "float32",
                "nodata":    np.nan,
                "count":     1,
            })

            self.logger.info(
                f"Clipped shape: {clipped.shape}, "
                f"valid pixels: {np.sum(~np.isnan(clipped))}"
            )

        # ── 3. Reproject to EPSG:4326 at target resolution ───────────────────
        
        new_transform, new_width, new_height, bounds = get_algeria_grid(boundary_path)

        elevation_resampled = np.full(
            (new_height, new_width), np.nan, dtype=np.float32
        )

        reproject(
            source=clipped,
            destination=elevation_resampled,
            src_transform=clip_transform,
            src_crs=clip_profile["crs"],
            dst_transform=new_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        self.logger.info(
            f"Resampled to {new_height}×{new_width} "
            f"at ~{target_res_m}m resolution"
        )

        # ── 4. Compute slope and aspect ───────────────────────────────────────
        self.logger.info("Computing slope and aspect")
        # Fill NaN with 0 temporarily for gradient computation
        elev_filled = np.where(np.isnan(elevation_resampled), 0, elevation_resampled)
        slope_deg, aspect_deg = compute_slope_aspect(elev_filled, target_res_m)

        # Restore NaN mask
        nan_mask = np.isnan(elevation_resampled)
        slope_deg[nan_mask]  = np.nan
        aspect_deg[nan_mask] = np.nan

        self.logger.info(
            f"Elevation: min={np.nanmin(elevation_resampled):.0f}m "
            f"max={np.nanmax(elevation_resampled):.0f}m"
        )
        self.logger.info(
            f"Slope: min={np.nanmin(slope_deg):.1f}° "
            f"max={np.nanmax(slope_deg):.1f}°"
        )

        # ── 5. Save all three bands ───────────────────────────────────────────
        out_profile = {
            "driver":    "GTiff",
            "dtype":     "float32",
            "crs":       "EPSG:4326",
            "transform": new_transform,
            "width":     new_width,
            "height":    new_height,
            "count":     1,
            "nodata":    np.nan,
            "compress":  "lzw",
        }

        for name, array in [
            ("elevation", elevation_resampled),
            ("slope",     slope_deg),
            ("aspect",    aspect_deg),
        ]:
            out_path = out_dir / f"{name}.tif"
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(array, 1)
            self.logger.info(f"Saved → {out_path}")


    def load(self):
        """Return dict of curated rasters: elevation, slope, aspect."""
        out_dir = Path(self.config["dem"]["paths"]["curated"])
        result  = {}
        for name in ["elevation", "slope", "aspect"]:
            path = out_dir / f"{name}.tif"
            if not path.exists():
                raise FileNotFoundError(f"Run curate() first — missing {path}")
            with rasterio.open(path) as src:
                result[name] = src.read(1)
        self.logger.info("Loaded elevation, slope, aspect arrays")
        return result


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["dem"]["paths"]

    source = DEMSource(config)
    arrays = source.run()

    print(f"\nDEM curation complete")
    for name, arr in arrays.items():
        valid = np.sum(~np.isnan(arr))
        print(f"   {name}: shape={arr.shape}, valid pixels={valid}")