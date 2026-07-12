"""
OSM Roads — Static Data Source

Ingestion:  confirms .pbf file exists
Curation:   reads road network from .pbf
            filters to major roads only
            reprojects to EPSG:4326
            computes distance-to-nearest-road raster
            saves:
              curated/roads/roads.gpkg         ← road network
              curated/roads/road_distance.tif  ← distance raster (km)
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource
from utils.grid import get_algeria_grid


class OSMSource(DataSource):

    def ingest(self, start_date=None, end_date=None):
        """Static — confirm .pbf file exists."""
        pbf_path = self.raw_dir / self.config["osm"]["filename"]

        if not pbf_path.exists():
            raise FileNotFoundError(
                f"OSM file not found: {pbf_path}\n"
                f"Download from https://download.geofabrik.de/africa/algeria.html"
            )
        self.logger.info(f"OSM file found: {pbf_path.name} "
                         f"({pbf_path.stat().st_size / 1e6:.1f} MB)")


    def curate(self):
        """
        Extract roads from .pbf, compute distance raster.
        Uses pyogrio to read from .pbf directly.
        """
        pbf_path      = self.raw_dir / self.config["osm"]["filename"]
        out_dir       = Path(self.config["osm"]["paths"]["curated"])
        boundary_path = Path(self.config["gadm"]["paths"]["curated"]) / "algeria_country.gpkg"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Load Algeria boundary ─────────────────────────────────────────────
        algeria = gpd.read_file(boundary_path).to_crs("EPSG:4326")
        bbox    = tuple(algeria.total_bounds)  # (minx, miny, maxx, maxy)

        # ── Read roads from .pbf ──────────────────────────────────────────────
        self.logger.info("Reading roads from OSM .pbf (this may take 2-5 minutes)")
        try:
            roads = gpd.read_file(
                pbf_path,
                layer="lines",
                bbox=bbox,
                engine="pyogrio"
            )
        except Exception as e:
            self.logger.error(f"Failed to read .pbf: {e}")
            self.logger.info("Trying alternative read method...")
            roads = gpd.read_file(f"/{pbf_path}", bbox=bbox)

        self.logger.info(f"Total road features: {len(roads)}")

        # ── Filter to major roads ─────────────────────────────────────────────
        # highway tag defines road type in OSM
        major_types = [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "motorway_link", "trunk_link", "primary_link", "secondary_link"
        ]

        if "highway" in roads.columns:
            roads = roads[roads["highway"].isin(major_types)].copy()
            self.logger.info(f"Major roads after filtering: {len(roads)}")
        else:
            self.logger.warning("No 'highway' column found — keeping all lines")

        # ── Clip to Algeria ───────────────────────────────────────────────────
        roads = roads.to_crs("EPSG:4326")
        roads = gpd.clip(roads, algeria)
        self.logger.info(f"Roads after clipping to Algeria: {len(roads)}")

        # ── Save road network ─────────────────────────────────────────────────
        roads_path = out_dir / "roads.gpkg"
        roads[["geometry", "highway"] if "highway" in roads.columns 
              else ["geometry"]].to_file(roads_path, driver="GPKG")
        self.logger.info(f"Saved road network → {roads_path}")

        # ── Compute distance-to-road raster ──────────────────────────────────
        self.logger.info("Computing distance-to-road raster")

        grid = get_algeria_grid(boundary_path, self.config["algeria"]["grid_resolution_m"])

        road_mask = rasterize(
            [(geom, 1) for geom in roads.geometry if geom is not None],
            out_shape=grid.shape(),
            transform=grid.transform,
            fill=0,
            dtype=np.uint8,
        )

        no_road_mask = (road_mask == 0)
        dist_pixels = distance_transform_edt(no_road_mask)
        dist_km = dist_pixels * (grid.res_m / 1000.0)

        self.logger.info(
            f"Distance range: {dist_km.min():.1f} → {dist_km.max():.1f} km"
        )

        # ── Save distance raster ──────────────────────────────────────────────
        dist_path = out_dir / "road_distance.tif"
        with rasterio.open(dist_path, "w", **{
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
            dst.write(dist_km, 1)

        self.logger.info(f"Saved distance raster → {dist_path}")


    def load(self):
        """Return road distance raster as numpy array."""
        path = Path(self.config["osm"]["paths"]["curated"]) / "road_distance.tif"
        if not path.exists():
            raise FileNotFoundError("Run curate() first")
        with rasterio.open(path) as src:
            arr = src.read(1)
        self.logger.info(f"Loaded road distance raster: {arr.shape}")
        return arr


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["osm"]["paths"]

    source = OSMSource(config)
    arr    = source.run()

    print(f"\nOSM curation complete")
    print(f"   Distance raster shape: {arr.shape}")
    print(f"   Distance range: {np.nanmin(arr):.1f} → {np.nanmax(arr):.1f} km")