"""
Pipeline Orchestrator

Training:    python orchestrator.py --mode train
Operational: python orchestrator.py --mode operational
"""

import argparse
import re
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
from ingest.landcover import LandCoverSource


def build_cfg(config: dict, key: str, mode: str) -> dict:
    """
    Build the configuration passed to a source.

    Static sources:
        use their normal raw/curated paths.

    Dynamic sources:
        train       -> raw_training / curated_training
        operational -> raw_operational / curated_operational

    The source classes always receive:
        paths["raw"]
        paths["curated"]
    """
    cfg = config.copy()

    source_config = config[key]
    paths = source_config["paths"].copy()

    if mode == "train":
        if "raw_training" in paths:
            paths["raw"] = paths.pop("raw_training")
        if "curated_training" in paths:
            paths["curated"] = paths.pop("curated_training")

    elif mode == "operational":
        if "raw_operational" in paths:
            paths["raw"] = paths.pop("raw_operational")
        if "curated_operational" in paths:
            paths["curated"] = paths.pop("curated_operational")

    else:
        raise ValueError(f"Unsupported pipeline mode: {mode}")

    # Every source must ultimately receive generic raw/curated paths.
    if "raw" not in paths:
        raise KeyError(
            f"{key}: no raw path configured for mode '{mode}'"
        )

    if "curated" not in paths:
        raise KeyError(
            f"{key}: no curated path configured for mode '{mode}'"
        )

    cfg["paths"] = paths

    # FIRMS uses different sensors for historical and operational data.
    if key == "firms":
        if mode == "train":
            cfg["sensors"] = config["firms"]["sensors"]
        else:
            cfg["sensors"] = config["firms"]["sensors_nrt"]

    return cfg


def choose_firms_operational_window(config: dict, today: datetime, fire_months):
    """
    - Use a 7-day refresh if all sensors have continuous coverage; otherwise backfill the full season.
    """
    season_start = today.replace(month=min(fire_months), day=1).date()
    path_config = config["firms"]["paths"]
    raw_value = path_config.get("raw_operational", path_config.get("raw"))
    raw_dir = Path(raw_value)
    sensors = {str(sensor).upper() for sensor in config["firms"].get("sensors_nrt", [])}
    pattern = re.compile(
        r"^firms_(?P<sensor>.+)_(?P<start>\d{4}-\d{2}-\d{2})_"
        r"(?P<end>\d{4}-\d{2}-\d{2})\.csv$"
    )

    intervals_by_sensor = {sensor: [] for sensor in sensors}
    for path in raw_dir.glob("firms_*.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        sensor = match.group("sensor").upper()
        if sensor not in intervals_by_sensor:
            continue
        try:
            start_date = datetime.strptime(match.group("start"), "%Y-%m-%d").date()
            end_date = datetime.strptime(match.group("end"), "%Y-%m-%d").date()
        except ValueError:
            continue
        intervals_by_sensor[sensor].append((start_date, end_date))

    required_end = today.date() - timedelta(days=1)
    continuous = True
    coverage_notes = []
    for sensor in sensors:
        intervals = sorted(intervals_by_sensor[sensor])
        cursor = season_start
        if not intervals:
            continuous = False
            coverage_notes.append(f"{sensor}: no retained raw files")
            continue
        for interval_start, interval_end in intervals:
            if interval_end < season_start:
                continue
            interval_start = max(interval_start, season_start)
            if interval_start > cursor:
                continuous = False
                coverage_notes.append(f"{sensor}: gap {cursor} → {interval_start - timedelta(days=1)}")
                break
            cursor = max(cursor, interval_end + timedelta(days=1))
            if cursor > required_end:
                break
        if cursor <= required_end:
            continuous = False
            coverage_notes.append(f"{sensor}: coverage ends {cursor - timedelta(days=1)}")

    if continuous and sensors:
        start = today.date() - timedelta(days=7)
        reason = "recent overlap refresh"
    else:
        start = season_start
        reason = "initial/current-season backfill"
        if coverage_notes:
            print("FIRMS archive incomplete; using full current-season backfill: " + "; ".join(coverage_notes))
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), reason


def preflight_check(config: dict, mode: str):
    import os
    errors = []

    if not os.getenv("FIRMS_MAP_KEY"):
        errors.append("FIRMS_MAP_KEY not set in .env")

    cds_rc = Path.home() / ".cdsapirc"
    if not cds_rc.exists():
        errors.append("ERA5-Land: ~/.cdsapirc not found — register at cds.climate.copernicus.eu")

    if not config["sentinel"].get("gee_project"):
        errors.append("Sentinel-2: sentinel.gee_project empty in config.yaml")

    if mode == "train":
        static_checks = [
            (config["gadm"]["paths"]["raw"],        config["gadm"]["filename_gpkg"],       "GADM"),
            (config["dem"]["paths"]["raw"],          config["dem"]["filename"],              "DEM"),
            (config["worldpop"]["paths"]["raw"],     config["worldpop"]["filename"],         "WorldPop"),
            (config["osm"]["paths"]["raw"],          config["osm"]["filename"],              "OSM"),
            # landcover downloads itself — no pre-existing file needed
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
    - Run one source with failure isolation.
    - A failed source logs the error but doesn't kill the whole run.
    - Returns True if successful, False if failed.
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

    # Date windows per mode 
    if mode == "train":
        start_date = config["training"]["start_date"]
        end_date   = config["training"]["end_date"]
    else:
        # Operational — per-source lag and window configurations
        today = datetime.today()
        
        fire_months = config["training"].get("fire_season_months", list(range(1, 13)))
        in_fire_season = today.month in fire_months

        # FIRMS needs the current-season archive for days_since_fire
        if in_fire_season:
            firms_start, firms_end, firms_window_reason = choose_firms_operational_window(
                config, today, fire_months
            )
        else:
            firms_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
            firms_end = today.strftime("%Y-%m-%d")
            firms_window_reason = "outside fire-season bounded refresh"

        # ERA5-Land: use the latest available date (approximately seven days
        # behind today). The current-season range is needed for recursive FWI
        # state and for the seven-day weather features.
        era5_end_dt = today - timedelta(days=config.get("operational", {}).get("era5_lag_days", 7))
        if in_fire_season:
            era5_start_dt = era5_end_dt.replace(month=min(fire_months), day=1)
        else:
            era5_start_dt = era5_end_dt - timedelta(days=7)
        era5_end = era5_end_dt.strftime("%Y-%m-%d")
        era5_start = era5_start_dt.strftime("%Y-%m-%d")

        # Sentinel-2: Fetch from start of the month to capture latest cloud-free satellite composite
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
        print(f"  FIRMS:    {firms_start} → {firms_end} (NRT; {firms_window_reason})")
        
        print(f"  ERA5-Land: {era5_start} → {era5_end} (ERA5T 9km)")
        
        print(f"  Sentinel: {sentinel_start} → {sentinel_end}")
    
    print(f"{'='*55}\n")

    preflight_check(config, mode)

    results = {}

    # Static sources — training/setup only 
    # Never re-run in operational mode 
    if mode == "train" or force_static:
        print("\nSTATIC SOURCES (run once)")

        # GADM must succeed — everything else depends on it
        gadm = GADMSource(build_cfg(config, "gadm", mode))
        ok = run_source("GADM Boundaries", gadm)
        if not ok:
            print("\nGADM failed — all other sources depend on it. Stopping.")
            sys.exit(1)

        results["dem"] = run_source(
            "DEM",
            DEMSource(build_cfg(config, "dem", mode))
        )

        results["worldpop"] = run_source(
            "WorldPop",
            WorldPopSource(build_cfg(config, "worldpop", mode))
        )

        results["osm"] = run_source(
            "OSM Roads",
            OSMSource(build_cfg(config, "osm", mode))
        )

        results["landcover"] = run_source(
            "Land Cover",
            LandCoverSource(build_cfg(config, "landcover", mode))
        )
    # Dynamic sources 
    print("\nDYNAMIC SOURCES")

    if mode == "train":
        results["firms"] = run_source(
            "FIRMS (historical)",
            FIRMSSource(build_cfg(config, "firms", mode)),
            start_date=start_date,
            end_date=end_date
        )

        results["era5"] = run_source(
            "ERA5",
            ERA5Source(build_cfg(config, "era5", mode)),
            start_date=start_date,
            end_date=end_date
        )

        results["sentinel"] = run_source(
            "Sentinel-2",
            SentinelSource(build_cfg(config, "sentinel", mode)),
            start_date=start_date,
            end_date=end_date
        )

    else:
        results["firms"] = run_source(
            "FIRMS (NRT)",
            FIRMSSource(build_cfg(config, "firms", mode)),
            start_date=firms_start,
            end_date=firms_end
        )

        results["era5"] = run_source(
            "ERA5 (ERA5T)",
            ERA5Source(build_cfg(config, "era5", mode)),
            start_date=era5_start,
            end_date=era5_end
        )

        results["sentinel"] = run_source(
            "Sentinel-2",
            SentinelSource(build_cfg(config, "sentinel", mode)),
            start_date=sentinel_start,
            end_date=sentinel_end
        )

    # Summary 
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