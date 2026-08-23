"""
Global Administrative Areas (GADM) — Static Data Source

Output files:
  curated/boundaries/algeria_country.gpkg   ( country outline )
  curated/boundaries/algeria_wilayas.gpkg   ( wilayas with names )
  curated/boundaries/algeria_communes.gpkg  ( communes with their names )
"""

import geopandas as gpd
from pathlib import Path
from loguru import logger
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource
import pandas as pd


class GADMSource(DataSource):

    def ingest(self, start_date=None, end_date=None):
        """
        - File was downloaded manually from gadm.org.
        - This step just confirms the file exists and is readable.
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
        - Extract admin levels 0, 1 and 2 from GADM GeoPackage
        - Reproject to EPSG:4326
        - Validate geometry and save.
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

            gdf = gdf.to_crs(self.config["algeria"]["crs"])

            invalid = (~gdf.geometry.is_valid).sum()
            if invalid > 0:
                self.logger.warning(f"{invalid} invalid geometries — attempting fix")
                gdf["geometry"] = gdf.geometry.buffer(0)

            keep_cols = ["geometry"]

            if level == 0:
                optional = ["GID_0", "COUNTRY"]
                required = ["GID_0"]  
            elif level == 1:
                optional = ["GID_0", "GID_1", "COUNTRY", "NAME_1", "TYPE_1"]
                required = ["GID_1", "NAME_1"]  
            elif level == 2:
                optional = ["GID_0", "GID_1", "GID_2", "COUNTRY", "NAME_1", "NAME_2", "TYPE_1", "TYPE_2"]
                required = ["GID_1", "GID_2", "NAME_2"]  

            # fail loud instead of silently dropping a required column
            missing_required = [c for c in required if c not in gdf.columns]
            if missing_required:
                raise ValueError(
                    f"Layer {layer_name} is missing required columns {missing_required} "
                    f"— downstream joins in build_dataset.py depend on these."
                )

            keep_cols += [c for c in optional if c in gdf.columns]
            gdf = gdf[keep_cols]

            id_col = {0: "GID_0", 1: "GID_1", 2: "GID_2"}.get(level)
            if id_col and id_col in gdf.columns:
                n_before = len(gdf)
                n_ids = gdf[id_col].nunique()
                if n_before > n_ids:
                    dup_ids = gdf[gdf[id_col].duplicated(keep=False)][id_col].unique()
                    self.logger.warning(
                        f"{n_before - n_ids} duplicate {id_col} rows found across {len(dup_ids)} IDs — "
                        f"checking whether geometries are identical or need merging: {list(dup_ids[:10])}"
                    )
                    # Check geometry equality within each duplicated ID
                    needs_dissolve = []
                    for uid in dup_ids:
                        geoms = gdf.loc[gdf[id_col] == uid, "geometry"]
                        if not all(g.equals(geoms.iloc[0]) for g in geoms):
                            needs_dissolve.append(uid)

                    if needs_dissolve:
                        self.logger.warning(
                            f"{len(needs_dissolve)} IDs have DIFFERING geometries under the same {id_col} — "
                            f"these are likely legitimate multipart records that were split across rows. "
                            f"Dissolving (union) rather than dropping: {needs_dissolve[:10]}"
                        )
                        # Dissolve only the problematic IDs, keep everything else as-is
                        dissolve_mask = gdf[id_col].isin(needs_dissolve)
                        dissolved = gdf[dissolve_mask].dissolve(by=id_col, as_index=False)
                        gdf = pd.concat([gdf[~dissolve_mask], dissolved], ignore_index=True)
                    else:
                        # True duplicates — identical geometry, safe to drop
                        gdf = gdf.drop_duplicates(subset=[id_col], keep="first")
                        self.logger.info(f"All duplicates were identical geometries — safely dropped to {len(gdf)} rows")

            if level == 0:
                out_path = out_dir / "algeria_country.gpkg"
                gdf.to_file(out_path, driver="GPKG")
                self.logger.info(f"Saved country boundary → {out_path}")

            elif level == 1:
                out_path = out_dir / "algeria_wilayas.gpkg"
                gdf.to_file(out_path, driver="GPKG")
                self.logger.info(f"Saved {len(gdf)} wilayas → {out_path}")
                if "NAME_1" in gdf.columns:
                    self.logger.info(f"Wilayas: {sorted(gdf['NAME_1'].tolist())}")

            elif level == 2:
                out_path = out_dir / "algeria_communes.gpkg"
                gdf.to_file(out_path, driver="GPKG")
                self.logger.info(f"Saved {len(gdf)} communes → {out_path}")
                if not (1500 <= len(gdf) <= 1600):
                    raise ValueError(
                        f"Unexpected commune count: {len(gdf)} (expected 1500-1600). "
                        f"GADM source file may be malformed, or Algeria's commune structure "
                        f"has changed — verify against current ONS data before proceeding."
                    )
                if "NAME_2" in gdf.columns:
                    self.logger.info(f"First 20 communes: {sorted(gdf['NAME_2'].tolist())[:20]}")


    def load(self, level=2):
        if level == 0:
            path = self.cur_dir / "algeria_country.gpkg"
        elif level == 1:
            path = self.cur_dir / "algeria_wilayas.gpkg"
        elif level == 2:
            path = self.cur_dir / "algeria_communes.gpkg"
        else:
            raise ValueError("level must be 0, 1 or 2")

        if not path.exists():
            raise FileNotFoundError("Run curate() first")

        return gpd.read_file(path)


# ── Run standalone (for testing) ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Inject source-specific paths into config for base class
    config["paths"] = config["gadm"]["paths"]

    source = GADMSource(config)

    source.run()
    wilayas = source.load(level=1)
    communes = source.load(level=2)
    print(f"Wilayas: {len(wilayas)}, Communes: {len(communes)}")