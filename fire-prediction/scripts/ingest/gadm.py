"""
GADM Algeria Boundaries — Static Data Source

Ingestion:  reads the .gpkg file 
Curation:   extracts admin level 0 (country) and level 1 (wilayas)
            reprojects to EPSG:4326
            validates geometry
            saves clean GeoPackages to curated/boundaries/

Output files:
  curated/boundaries/algeria_country.gpkg   ← country outline
  curated/boundaries/algeria_wilayas.gpkg   ← 48 wilayas with names
"""

import geopandas as gpd
from pathlib import Path
from loguru import logger
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource


class GADMSource(DataSource):

    def ingest(self, start_date=None, end_date=None):
        """
        Static source — no download needed.
        File was downloaded manually from gadm.org.
        This step just confirms the file exists and is readable.
        """
        gpkg = self.raw_dir / self.config["gadm"]["filename_gpkg"]
        
        if not gpkg.exists():
            raise FileNotFoundError(
                f"GADM file not found: {gpkg}\n"
                f"Download from https://gadm.org/download_country.html → Algeria"
            )
        
        # Confirm it's readable
        layers = gpd.list_layers(gpkg)
        self.logger.info(f"Found GADM file: {gpkg.name}")
        self.logger.info(f"Available layers: {layers['name'].tolist()}")


    def curate(self):
        """
        Extract admin levels 0 and 1 from GADM GeoPackage.
        Reproject to EPSG:4326.
        Validate and save.
        """
        gpkg = self.raw_dir / self.config["gadm"]["filename_gpkg"]
        out_dir = self.cur_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for level in self.config["gadm"]["admin_levels"]:
            layer_name = f"ADM_ADM_{level}"
            self.logger.info(f"Processing admin level {level} ({layer_name})")

            try:
                gdf = gpd.read_file(gpkg, layer=layer_name)
            except Exception as e:
                self.logger.warning(f"Layer {layer_name} not found: {e}")
                continue

            # Reproject to standard CRS
            gdf = gdf.to_crs(self.config["algeria"]["crs"])

            # Validate geometry
            invalid = (~gdf.geometry.is_valid).sum()
            if invalid > 0:
                self.logger.warning(f"{invalid} invalid geometries — attempting fix")
                gdf["geometry"] = gdf.geometry.buffer(0)

            # Select only useful columns
            keep_cols = ["geometry"]
            optional = ["GID_0", "GID_1", "COUNTRY", "NAME_1", "TYPE_1"]
            keep_cols += [c for c in optional if c in gdf.columns]
            gdf = gdf[keep_cols]

            # Save
            if level == 0:
                out_path = out_dir / "algeria_country.gpkg"
                gdf.to_file(out_path, driver="GPKG")
                self.logger.info(f"Saved country boundary → {out_path}")

            elif level == 1:
                out_path = out_dir / "algeria_wilayas.gpkg"
                gdf.to_file(out_path, driver="GPKG")
                self.logger.info(
                    f"Saved {len(gdf)} wilayas → {out_path}"
                )
                # Print wilaya names as sanity check
                if "NAME_1" in gdf.columns:
                    self.logger.info(
                        f"Wilayas: {sorted(gdf['NAME_1'].tolist())}"
                    )


    def load(self):
        """Return curated wilayas as GeoDataFrame."""
        path = self.cur_dir / "algeria_wilayas.gpkg"
        if not path.exists():
            raise FileNotFoundError("Run curate() first")
        return gpd.read_file(path)


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Inject source-specific paths into config for base class
    config["paths"] = config["gadm"]["paths"]

    source = GADMSource(config)

    wilayas = source.run()

    print(f"\n GADM curation complete")
    print(f"   Wilayas loaded: {len(wilayas)}")
    print(f"   Columns: {list(wilayas.columns)}")
    print(f"   CRS: {wilayas.crs}")
    print(wilayas[["NAME_1", "geometry"]].head())