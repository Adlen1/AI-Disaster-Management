"""
WorldPop Population Density — Static Data Source

Output files:
    curated/worldpop/population_density.tif
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource
from utils.grid import get_algeria_grid


class WorldPopSource(DataSource):

    def ingest(self, start_date=None, end_date=None):
        """ 
        - Confirms file exists
        """
        pop_path = self.raw_dir / self.config["worldpop"]["filename"]

        if not pop_path.exists():
            raise FileNotFoundError(
                f"WorldPop file not found: {pop_path}\n"
                f"Download from https://hub.worldpop.org/geodata/summary?id=44981"
            )

        with rasterio.open(pop_path) as src:
            self.logger.info(f"WorldPop file: {pop_path.name}")
            self.logger.info(f"  CRS:    {src.crs}")
            self.logger.info(f"  Shape:  {src.shape}")
            self.logger.info(f"  NoData: {src.nodata}")


    def curate(self):
        """
        - Clips to Algeria
        - Reprojects to EPSG:4326
        - Resamples to 1km (already 1km, but aligns to DEM grid)
        - Replaces nodata with NaN
        """
        pop_path      = self.raw_dir / self.config["worldpop"]["filename"]
        out_dir       = Path(self.config["worldpop"]["paths"]["curated"])
        boundary_path = Path(self.config["gadm"]["paths"]["curated"]) / "algeria_country.gpkg"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load boundary
        algeria = gpd.read_file(boundary_path).to_crs("EPSG:4326")

        # Clip to Algeria
        self.logger.info("Clipping WorldPop to Algeria")
        with rasterio.open(pop_path) as src:
            algeria_src_crs = algeria.to_crs(src.crs)
            clip_shapes     = [g.__geo_interface__ for g in algeria_src_crs.geometry]

            clipped, clip_transform = mask(
                src, clip_shapes, crop=True, nodata=-99999
            )
            clipped = clipped[0].astype(np.float32)
            clipped[clipped < 0] = np.nan  # replace nodata with NaN

            src_crs = src.crs

        self.logger.info(
            f"Clipped — valid pixels: {np.sum(~np.isnan(clipped))}"
        )
        self.logger.info(
            f"Population range: "
            f"{np.nanmin(clipped):.1f} → {np.nanmax(clipped):.1f} people/km²"
        )

        # Reproject and align to 1km grid
        
        grid = get_algeria_grid(boundary_path, self.config["algeria"]["grid_resolution_m"])

        pop_resampled = grid.empty_array()

        reproject(
            source=clipped,
            destination=pop_resampled,
            src_transform=clip_transform,
            src_crs=src_crs,
            dst_transform=grid.transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        # Save
        out_path = out_dir / "population_density.tif"
        with rasterio.open(out_path, "w", **{
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
            dst.write(pop_resampled, 1)

        self.logger.info(f"Saved → {out_path}")
        self.logger.info(
            f"Shape: {pop_resampled.shape}, "
            f"valid: {np.sum(~np.isnan(pop_resampled))} pixels"
        )
        

    def load(self):
        """Return curated population density array."""
        path = Path(self.config["worldpop"]["paths"]["curated"]) / "population_density.tif"
        if not path.exists():
            raise FileNotFoundError("Run curate() first")
        with rasterio.open(path) as src:
            arr = src.read(1)
        self.logger.info(f"Loaded population density: {arr.shape}")
        return arr


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["worldpop"]["paths"]

    source = WorldPopSource(config)
    arr    = source.run()

    print(f"\nWorldPop curation complete")
    print(f"   Shape:  {arr.shape}")
    print(f"   Valid pixels: {np.sum(~np.isnan(arr))}")
    print(f"   Population range: {np.nanmin(arr):.1f} → {np.nanmax(arr):.1f} people/km²")