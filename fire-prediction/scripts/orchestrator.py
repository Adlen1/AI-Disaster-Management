# scripts/orchestrator.py
"""
Pipeline Orchestrator

Training:    python orchestrator.py --mode train
Operational: python orchestrator.py --mode operational

Static sources run only in train mode (or forced with --force-static).
Dynamic sources run in both modes with different date windows.
"""

import argparse
import sys
import traceback
import yaml
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from utils.logger    import setup_logger
from ingest.gadm     import GADMSource
from ingest.dem      import DEMSource
from ingest.worldpop import WorldPopSource
from ingest.osm      import OSMSource
from ingest.firms    import FIRMSSource
from ingest.era5     import ERA5Source
from ingest.sentinel import SentinelSource


def build_cfg(config: dict, key: str) -> dict:
    cfg = config.copy()
    cfg["paths"] = config[key]["paths"]
    return cfg


def preflight_check(config: dict, mode: str):
    """
    Verify credentials and required files exist before starting.
    """
    import os
    errors = []

    # FIRMS key
    if not os.getenv("FIRMS_MAP_KEY"):
        errors.append("FIRMS_MAP_KEY not set in .env")

    # CDS API for ERA5
    cds_rc = Path.home() / ".cdsapirc"
    if not cds_rc.exists():
        errors.append(f"ERA5: ~/.cdsapirc not found — register at cds.climate.copernicus.eu")

    # GEE project
    if not config["sentinel"].get("gee_project"):
        errors.append("Sentinel-2: sentinel.gee_project empty in config.yaml")

    # Static files (only if we're going to run static sources)
    if mode == "train":
        static_checks = [
            (config["gadm"]["paths"]["raw"],     config["gadm"]["filename_gpkg"],    "GADM"),
            (config["dem"]["paths"]["raw"],      config["dem"]["filename"],           "DEM"),
            (config["worldpop"]["paths"]["raw"], config["worldpop"]["filename"],      "WorldPop"),
            (config["osm"]["paths"]["raw"],      config["osm"]["filename"],           "OSM"),
        ]
        for raw_dir, filename, name in static_checks:
            path = Path(raw_dir) / filename
            if not path.exists():
                errors.append(f"{name}: raw file not found at {path}")

    if errors:
        print("\nPre-flight check failed:\n")
        for e in errors:
            print(f"   • {e}")
        print()
        sys.exit(1)

    print("Pre-flight check passed\n")


def run_source(name: str, source, start_date=None, end_date=None):
    """
    Run one source with failure isolation.
    A failed source logs the error but doesn't kill the whole run.
    Returns True if successful, False if failed.
    """
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    try:
        source.run(start_date=start_date, end_date=end_date)
        print(f"  Success {name} complete")
        return True
    except Exception as e:
        print(f"  Failed {name} failed: {e}")
        traceback.print_exc()
        return False


def run_pipeline(mode: str, force_static: bool = False):
    logger = setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # ── Date windows per mode ─────────────────────────────────────────────────
    if mode == "train":
        start_date = config["training"]["start_date"]
        end_date   = config["training"]["end_date"]
    else:
        # Operational — per-source lag to account for data availability delays
        # ERA5 preliminary (ERA5T) lags ~5 days
        # Sentinel-2 revisit + processing ~5 days
        # FIRMS NRT available within ~3 hours
        today = datetime.today()
        firms_start = firms_end = today.strftime("%Y-%m-%d")

        era5_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        era5_start = era5_end = era5_date

        sentinel_start = today.replace(day=1).strftime("%Y-%m-%d")
        sentinel_end = today.strftime("%Y-%m-%d")

    print(f"\n{'='*55}")
    print(f"  Pipeline — mode: {mode.upper()}")
    if mode == "train":
        print(f"  Dates: {start_date} → {end_date}")
        if config["training"]["fire_season_only"]:
            months = config["training"]["fire_season_months"]
            print(f"  Fire season only: {months}")
    else:
        print(f"  FIRMS:    {firms_start} (NRT)")
        print(f"  ERA5:     {era5_start} → {era5_start_end} (ERA5T)")
        print(f"  Sentinel: {sentinel_start} → {sentinel_start_end}")
    print(f"{'='*55}\n")

    preflight_check(config, mode)

    results = {}

    # ── Static sources — training/setup only ──────────────────────────────────
    # Never re-run in operational mode — they never change day-to-day
    if mode == "train" or force_static:
        print("\nSTATIC SOURCES (run once)")

        # GADM must succeed — everything else depends on it
        gadm = GADMSource(build_cfg(config, "gadm"))
        ok = run_source("GADM Boundaries", gadm)
        if not ok:
            print("\nGADM failed — all other sources depend on it. Stopping.")
            sys.exit(1)

        results["dem"]      = run_source("DEM",      DEMSource(build_cfg(config, "dem")))
        results["worldpop"] = run_source("WorldPop", WorldPopSource(build_cfg(config, "worldpop")))
        results["osm"]      = run_source("OSM Roads",OSMSource(build_cfg(config, "osm")))

    # ── Dynamic sources ───────────────────────────────────────────────────────
    print("\nDYNAMIC SOURCES")

    if mode == "train":
        # Training — all use _SP (standard processing, science quality)
        # FIRMS: curate() reads pre-downloaded historical CSVs
        results["firms"] = run_source(
            "FIRMS (historical)",
            FIRMSSource(build_cfg(config, "firms")),
            start_date=start_date,
            end_date=end_date
        )
        results["era5"] = run_source(
            "ERA5",
            ERA5Source(build_cfg(config, "era5")),
            start_date=start_date,
            end_date=end_date
        )
        results["sentinel"] = run_source(
            "Sentinel-2",
            SentinelSource(build_cfg(config, "sentinel")),
            start_date=start_date,
            end_date=end_date
        )

    else:
        # Operational — FIRMS uses NRT sensors (available within ~3h)
        # ERA5 and Sentinel use recent window with appropriate lag
        op_config = config.copy()
        op_config["firms"] = config["firms"].copy()
        op_config["firms"]["sensors"] = [
            "MODIS_NRT",
            "VIIRS_SNPP_NRT",
            "VIIRS_NOAA20_NRT",
            "VIIRS_NOAA21_NRT",
        ]

        results["firms"] = run_source(
            "FIRMS (NRT)",
            FIRMSSource({**build_cfg(op_config, "firms")}),
            start_date=firms_start,
            end_date=firms_end
        )
        results["era5"] = run_source(
            "ERA5 (ERA5T)",
            ERA5Source(build_cfg(config, "era5")),
            start_date=era5_start,
            end_date=era5_start_end
        )
        results["sentinel"] = run_source(
            "Sentinel-2",
            SentinelSource(build_cfg(config, "sentinel")),
            start_date=sentinel_start,
            end_date=sentinel_start_end
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Run summary — mode: {mode.upper()}")
    print(f"{'='*55}")
    for name, ok in results.items():
        icon = "Success" if ok else "Failed"
        print(f"  {icon} {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n  Warning  {len(failed)} source(s) failed: {failed}")
        print(f"  Fix errors above and re-run — existing curated files are preserved.")
    else:
        print(f"\n  All sources complete.")
        if mode == "train":
            print(f"  Next: python scripts/integrate/build_dataset.py")
        else:
            print(f"  Next: python scripts/predict/run_prediction.py")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildfire prediction data pipeline")
    parser.add_argument(
        "--mode", choices=["train", "operational"],
        default="train",
        help="train = full historical pipeline | operational = today's prediction"
    )
    parser.add_argument(
        "--force-static", action="store_true",
        help="Re-run static sources even in operational mode (e.g. after OSM update)"
    )
    args = parser.parse_args()
    run_pipeline(mode=args.mode, force_static=args.force_static)