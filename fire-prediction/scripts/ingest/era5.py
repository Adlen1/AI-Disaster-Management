# scripts/ingest/era5.py
"""
Meteorological Reanalysis (ERA5) — Dynamic Data Source

Output files:
    raw/era5/era5_YYYYMM.nc        
    curated/era5/era5_YYYYMMDD_YYYYMMDD.parquet
"""
import shutil
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import cdsapi
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from shapely.geometry import Point

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.base import DataSource
from utils.base import filter_fire_season


# ── FWI VECTORIZED COMPUTATION ─────────────────────────────
# Canadian Fire Weather Index — Van Wagner (1987)
# Optimized with full vector calculations across spatial grid cell indices.

def compute_relative_humidity(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    """
    - Derive RH (%) from temperature and dewpoint (Kelvin).
    """
    temp_c = temp_k - 273.15
    dew_c  = dewpoint_k - 273.15
    rh = 100 * (
        np.exp((17.625 * dew_c)  / (243.04 + dew_c)) /
        np.exp((17.625 * temp_c) / (243.04 + temp_c))
    )
    return np.clip(rh, 0.0, 100.0)


def compute_wind_speed_kmh(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    - Wind speed in km/h from ERA5 U/V components (m/s). FWI needs km/h.
    """
    return np.sqrt(u**2 + v**2) * 3.6


def compute_wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    - Wind direction in degrees (meteorological convention, 0=North).
    """
    return (270 - np.degrees(np.arctan2(v, u))) % 360


def _ffmc_vectorized(temp: np.ndarray, rh: np.ndarray, wind_kmh: np.ndarray, rain: np.ndarray, prev_ffmc: np.ndarray) -> np.ndarray:
    """
    - Vectorized Fine Fuel Moisture Code across all active cells.
    """
    rh       = np.clip(rh, 0.0, 100.0)
    wind_kmh = np.maximum(0.0, wind_kmh)
    rain     = np.maximum(0.0, rain)

    mo = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)

    # Wetting Phase (precipitation > 0.5 mm)
    rain_mask = rain > 0.5
    rf = rain - 0.5
    mr = mo.copy()
    
    if np.any(rain_mask):
        mo_m = mo[rain_mask]
        rf_m = rf[rain_mask]
        mr_val = mo_m + 42.5 * rf_m * np.exp(-100.0 / (251.0 - mo_m)) * (1.0 - np.exp(-6.93 / rf_m))
        
        high_moist = mo_m > 150.0
        mr_val = np.where(high_moist, mr_val + 0.0015 * ((mo_m - 150.0)**2) * np.sqrt(rf_m), mr_val)
        mr[rain_mask] = np.minimum(mr_val, 250.0)

    # Equilibrium Moisture Content (EMC) Calculations
    ed = (0.942 * (rh**0.679)
          + 11.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))

    ew = (0.618 * (rh**0.753)
          + 10.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))

    # Drying Phase (moisture > ed)
    ko = (0.424 * (1.0 - (rh / 100.0)**1.7)
          + 0.0694 * np.sqrt(wind_kmh) * (1.0 - (rh / 100.0)**8))
    kd = ko * 0.581 * np.exp(0.0365 * temp)
    m_dry = ed + (mr - ed) * (10.0**(-kd))

    # Wetting Phase (moisture < ew)
    kl = (0.424 * (1.0 - ((100.0 - rh) / 100.0)**1.7)
          + 0.0694 * np.sqrt(wind_kmh) * (1.0 - ((100.0 - rh) / 100.0)**8))
    kw = kl * 0.581 * np.exp(0.0365 * temp)
    m_wet = ew - (ew - mr) * (10.0**(-kw))

    # Select appropriate drying/wetting values, keeping current if within the hysteresis envelope
    mo_final = np.where(mr > ed, m_dry, np.where(mr < ew, m_wet, mr))
    ffmc = 59.5 * (250.0 - mo_final) / (147.2 + mo_final)
    return np.clip(ffmc, 0.0, 101.0)


def _dmc_vectorized(temp: np.ndarray, rh: np.ndarray, rain: np.ndarray, month: int, prev_dmc: np.ndarray) -> np.ndarray:
    """
    - Vectorized Duff Moisture Code incorporating Mediterranean Day-Lengths.
    """
    rh   = np.clip(rh, 0.0, 100.0)
    rain = np.maximum(0.0, rain)

    # Wetting Phase (precipitation > 1.5 mm)
    rain_mask = rain > 1.5
    p_dmc = prev_dmc.copy()

    if np.any(rain_mask):
        re = 0.92 * rain[rain_mask] - 1.27
        mo = 20.0 + np.exp(5.6348 - prev_dmc[rain_mask] / 43.43)
        p_dmc_m = prev_dmc[rain_mask]
        
        # Piecewise calculation of b
        b = np.where(p_dmc_m <= 33.0, 
                     100.0 / (0.5 + 0.3 * p_dmc_m),
                     np.where(p_dmc_m <= 65.0, 
                              14.0 - 1.3 * np.log(p_dmc_m), 
                              6.2 * np.log(p_dmc_m) - 17.2))
                              
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * np.log(np.maximum(mr - 20.0, 1e-5))
        p_dmc[rain_mask] = np.maximum(pr, 0.0)

    # Drying Phase 
    algeria_dmc_le = [6.5, 7.5, 9.0, 12.8,13.9, 13.9, 12.4, 10.9,9.4, 8.0, 7.0, 6.0]
    
    month = np.clip(month, 1, 12)
    Le = algeria_dmc_le[month - 1]

    temp = np.maximum(temp, -1.1)

    k = (
        1.894
        * (temp + 1.1)
        * (100.0 - rh)
        * Le
        * 1e-6
    )

    dmc_new = p_dmc + 100.0 * k
    return np.maximum(dmc_new, 0.0)


def _dc_vectorized(temp: np.ndarray, rain: np.ndarray, month: int, prev_dc: np.ndarray) -> np.ndarray:
    """
    - Vectorized Drought Code incorporating Mediterranean Drying Factors.
    """
    rain = np.maximum(0.0, rain)

    # Wetting Phase (precipitation > 2.8 mm)
    rain_mask = rain > 2.8
    p_dc = prev_dc.copy()

    if np.any(rain_mask):
        rd = 0.83 * rain[rain_mask] - 1.27
        qo = 800.0 * np.exp(-prev_dc[rain_mask] / 400.0)
        qr = qo + 3.937 * rd
        dr = 400.0 * np.log(800.0 / np.maximum(qr, 1e-5))
        p_dc[rain_mask] = np.maximum(dr, 0.0)

    # Drying Phase (Algerian latitudes: 30°N to 40°N standard)
    algeria_dc_lf = [-1.6,-1.6,-1.6,0.9,3.8,5.8,6.4,5.0,2.4,0.4,-1.6,-1.6]
    Lf = algeria_dc_lf[month - 1]

    v = np.where(temp <= -2.8, Lf, 0.36 * (temp + 2.8) + Lf)
    v = np.maximum(v, 0.0)
    
    return np.maximum(p_dc + 0.5 * v, 0.0)


def _isi_vectorized(wind_kmh: np.ndarray, ffmc_val: np.ndarray) -> np.ndarray:
    """
    - Vectorized Initial Spread Index.
    """
    wind_kmh = np.maximum(0.0, wind_kmh)
    m = 147.2 * (101.0 - ffmc_val) / (59.5 + ffmc_val)
    m = np.maximum(m, 0.0)

    f_wind = np.exp(0.05039 * wind_kmh)
    f_fuel = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m**5.31) / 49300000.0)
    return 0.208 * f_wind * f_fuel


def _bui_vectorized(dmc_val: np.ndarray, dc_val: np.ndarray) -> np.ndarray:
    """
    - Vectorized Buildup Index explicitly resolving edge cases.
    """
    dmc_val = np.maximum(dmc_val, 0.0)
    dc_val  = np.maximum(dc_val, 0.0)

    denom = dmc_val + 0.4 * dc_val
    
    bui_under = np.where(denom > 0.0, 0.8 * dmc_val * dc_val / denom, 0.0)
    bui_over = dmc_val - (1.0 - 0.8 * dc_val / denom) * (0.92 + (0.0114 * dmc_val) ** 1.7)

    bui = np.where(dmc_val <= 0.4 * dc_val, bui_under, bui_over)
    bui = np.where((dmc_val == 0.0) & (dc_val == 0.0), 0.0, bui)
    return np.maximum(bui, 0.0)


def _fwi_vectorized(isi_val: np.ndarray, bui_val: np.ndarray) -> np.ndarray:
    """
    - Vectorized Fire Weather Index without fractional power warnings.
    """
    isi_val = np.maximum(isi_val, 0.0)
    bui_val = np.maximum(bui_val, 0.0)

    fd = np.where(bui_val <= 80.0,
                  0.626 * (bui_val**0.809) + 2.0,
                  1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui_val)))

    b = 0.1 * isi_val * fd
    
    # 1. To prevent log(0) issues, floor b at a small epsilon
    b_safe = np.maximum(b, 1e-5)
    
    # 2. To prevent raising negative numbers to the 0.647 fractional power,
    # we clamp the logarithmic term at a minimum of 0.0. 
    # (Since 0.434 * log(B) is only > 0 when B > 1.0, this is mathematically identical)
    log_term = np.maximum(0.434 * np.log(b_safe), 0.0)
    fwi_over_1 = np.exp(2.72 * (log_term**0.647))

    return np.where(b <= 0.0, 0.0, np.where(b > 1.0, fwi_over_1, b))


DEFAULT_FFMC = 85.0
DEFAULT_DMC = 6.0
DEFAULT_DC = 15.0
RESET_GAP_DAYS = 35


def compute_fwi_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Compute FWI components vectorized across all ERA5 grid cells.
    - Vectorized across all ERA5 cells for each day.
    - Loops only over dates because FFMC/DMC/DC are recursive.
    - Automatically resets moisture codes after a long gap
      (e.g. October -> May when only fire-season months are processed).
    """

    df = df.sort_values(["date", "era5_cell_id"]).copy()

    if df.duplicated(["date", "era5_cell_id"]).any():
        raise ValueError(
            "Duplicate (date, era5_cell_id) rows found before FWI computation."
        )

    cell_ids = np.sort(df["era5_cell_id"].unique())
    n_cells = len(cell_ids)

    cell_idx = pd.Series(
        np.arange(n_cells, dtype=np.int32),
        index=cell_ids,
    )

    prev_ffmc = np.full(n_cells, DEFAULT_FFMC, dtype=np.float64)
    prev_dmc = np.full(n_cells, DEFAULT_DMC, dtype=np.float64)
    prev_dc = np.full(n_cells, DEFAULT_DC, dtype=np.float64)

    # Track when each ERA5 cell was last processed
    last_seen = np.full(n_cells, np.datetime64("NaT"), dtype="datetime64[D]")

    results = []
    resets_applied = 0

    for date, day_df in df.groupby("date", sort=True):

        current_date = np.datetime64(pd.Timestamp(date).date())

        day_df = day_df.copy()

        idx = cell_idx.loc[day_df["era5_cell_id"]].to_numpy()

        # Reset cells whose previous observation is too far away
        # (i.e. crossed the skipped off-season)
        seen = ~np.isnat(last_seen[idx])

        if np.any(seen):
            gap_days = (
                current_date.astype("datetime64[D]")
                - last_seen[idx][seen]
            ).astype(int)

            reset_mask = np.zeros(len(idx), dtype=bool)
            reset_mask[seen] = gap_days > RESET_GAP_DAYS

            if np.any(reset_mask):
                prev_ffmc[idx[reset_mask]] = DEFAULT_FFMC
                prev_dmc[idx[reset_mask]] = DEFAULT_DMC
                prev_dc[idx[reset_mask]] = DEFAULT_DC
                resets_applied += reset_mask.sum()

        temp = day_df["temp_c"].to_numpy()
        rh = day_df["rh"].to_numpy()
        wind = day_df["wind_speed_kmh"].to_numpy()
        rain = day_df["precip_mm"].to_numpy()
        month = pd.Timestamp(date).month

        ffmc = _ffmc_vectorized(
            temp,
            rh,
            wind,
            rain,
            prev_ffmc[idx],
        )

        dmc = _dmc_vectorized(
            temp,
            rh,
            rain,
            month,
            prev_dmc[idx],
        )

        dc = _dc_vectorized(
            temp,
            rain,
            month,
            prev_dc[idx],
        )

        isi = _isi_vectorized(
            wind,
            ffmc,
        )

        bui = _bui_vectorized(
            dmc,
            dc,
        )

        fwi = _fwi_vectorized(
            isi,
            bui,
        )

        # Persist state
        prev_ffmc[idx] = ffmc
        prev_dmc[idx] = dmc
        prev_dc[idx] = dc

        last_seen[idx] = current_date

        day_df["FFMC"] = ffmc
        day_df["DMC"] = dmc
        day_df["DC"] = dc
        day_df["ISI"] = isi
        day_df["BUI"] = bui
        day_df["FWI"] = fwi

        results.append(day_df)

    print(f"Season resets applied: {resets_applied}")

    return pd.concat(results, ignore_index=True)


# ERA5 SOURCE 

class ERA5Source(DataSource):

    CDS_VARIABLES = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation",
        "volumetric_soil_water_layer_1",
    ]

    def ingest(self, start_date: str = None, end_date: str = None):
        """
        - Download ERA5 monthly NetCDF files for Algeria.
        """
        if not start_date:
            start_date = datetime.today().strftime("%Y-%m-%d")
        if not end_date:
            end_date = start_date

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        bbox = self.config["algeria"]["bbox"]

        try:
            c = cdsapi.Client()
        except Exception as e:
            raise RuntimeError(
                f"CDS API client failed: {e}\n"
                f"Make sure ~/.cdsapirc exists with your token.\n"
                f"See: https://cds.climate.copernicus.eu/ -> My Profile -> Token"
            )

        self.logger.info(f"ERA5 ingestion: {start_date} -> {end_date}")

        fire_months = self.config.get("training", {}).get(
            "fire_season_months", list(range(1, 13))
        )

        current = start.replace(day=1)
        while current <= end:
            year_str = str(current.year)
            month_str = f"{current.month:02d}"

            if current.month not in fire_months:
                self.logger.info(f"Skipping {year_str}-{month_str} (not fire season)")
                current += relativedelta(months=1)
                continue

            out_file = self.raw_dir / f"era5_{year_str}{month_str}.nc"

            if out_file.exists() and not zipfile.is_zipfile(out_file):
                self.logger.info(f"Already exists — skipping {out_file.name}")
                current += relativedelta(months=1)
                continue

            next_month = current + relativedelta(months=1)
            last_day = (next_month - timedelta(days=1)).day
            days = [f"{d:02d}" for d in range(1, last_day + 1)]
            all_hours = [f"{h:02d}:00" for h in range(24)]

            self.logger.info(f"Downloading ERA5: {year_str}-{month_str}")
            try:
                c.retrieve(
                    "reanalysis-era5-land",
                    {
                        "product_type": "reanalysis",
                        "variable": self.CDS_VARIABLES,
                        "year": year_str,
                        "month": month_str,
                        "day": days,
                        "time": all_hours,
                        "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
                        "data_format": "netcdf",
                        "download_format": "unarchived",
                    },
                    str(out_file)
                )
                self.logger.info(f"Saved → {out_file.name}")
                self._unwrap_if_zip(out_file)

            except Exception as e:
                self.logger.error(f"Failed {year_str}-{month_str}: {e}")

            current += relativedelta(months=1)

    def _unwrap_if_zip(self, out_file: Path):
        """Unwrap Zip archive if returned by Copernicus."""
        if not zipfile.is_zipfile(out_file):
            return

        self.logger.warning(f"{out_file.name} came back as zip — unwrapping")
        tmp_dir = out_file.parent / f"_unzip_{out_file.stem}"
        tmp_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(out_file, "r") as z:
                z.extractall(tmp_dir)

            nc_inside = list(tmp_dir.glob("*.nc"))
            if not nc_inside:
                raise RuntimeError(f"Zip had no .nc inside: {list(tmp_dir.iterdir())}")

            if len(nc_inside) == 1:
                shutil.move(str(nc_inside[0]), str(out_file))
            else:
                self.logger.info(f"Merging {len(nc_inside)} files: {[f.name for f in nc_inside]}")
                datasets = [self._standardize_era5_dataset(xr.open_dataset(f)) for f in nc_inside]
                merged = xr.merge(datasets, compat="override", join="outer")
                merged.to_netcdf(out_file)
                merged.close()
                for d in datasets:
                    d.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _standardize_era5_dataset(ds: xr.Dataset) -> xr.Dataset:
        """Resolve ERA5 coordinate naming and expver version merges."""
        if "time" not in ds.coords and "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})

        if "expver" in ds.dims:
            expver_vals = list(ds.expver.values)
            if 1 in expver_vals and 5 in expver_vals:
                ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
            else:
                ds = ds.isel(expver=0)
            ds = ds.drop_vars("expver", errors="ignore")

        return ds

    def _open_era5_file(self, nc_file: Path) -> xr.Dataset:
        if zipfile.is_zipfile(nc_file):
            raise RuntimeError(
                f"{nc_file.name} is still a zip. Delete and re-run ingest()."
            )
        ds = xr.open_dataset(nc_file, engine="netcdf4")
        ds = self._standardize_era5_dataset(ds)

        if "time" not in ds.coords:
            raise RuntimeError(
                f"{nc_file.name} has no valid time coordinates. Coords: {list(ds.coords)}."
            )
        return ds

    def curate(self):
        """
        - Process monthly NetCDF files into native curated Parquet rows.
        - Handles: in_algeria mask, precipitation accumulation fix,
        NaN border cells, FWI computation, fire season filter.
        """
        out_dir = Path(self.config["era5"]["paths"]["curated"])
        boundary_path = Path(self.config["gadm"]["paths"]["curated"]) / "algeria_country.gpkg"
        out_dir.mkdir(parents=True, exist_ok=True)

        nc_files = sorted(self.raw_dir.glob("era5_*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No ERA5 .nc files in {self.raw_dir}. Run ingest() first.")

        self.logger.info(f"Processing {len(nc_files)} ERA5 monthly files")

        algeria = gpd.read_file(boundary_path).to_crs("EPSG:4326")
        algeria_geom = algeria.geometry.union_all()

        # COMPUTE in_algeria MASK ONCE 
        first_ds = self._open_era5_file(nc_files[0])
        lats = first_ds.latitude.values
        lons = first_ds.longitude.values
        first_ds.close()

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        flat_lons = lon_grid.flatten()
        flat_lats = lat_grid.flatten()

        points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(flat_lons, flat_lats),
            crs="EPSG:4326"
        )
        in_algeria = points.within(algeria_geom).values
        cell_lons  = flat_lons[in_algeria]
        cell_lats  = flat_lats[in_algeria]
        cell_ids   = np.where(in_algeria)[0]

        self.logger.info(
            f"ERA5 cells inside Algeria: {len(cell_ids)} (from {len(flat_lons)} total)"
        )

        all_months = []

        for nc_file in nc_files:
            self.logger.info(f"Processing {nc_file.name}")
            ds = self._open_era5_file(nc_file)

            times = pd.to_datetime(ds.time.values)
            dates = pd.DatetimeIndex(np.unique(times.date))

            for date in dates:
                date_str   = str(date.date())
                date_mask  = times.date == date.date()
                day_ds     = ds.isel(time=date_mask)
                day_times  = times[date_mask]

                #  PRECIPITATION: max of daily accumulation (ERA5-Land tp is a running accumulation from 00:00 UTC)
                if "tp" in ds:
                    precip_daily = day_ds["tp"].values.max(axis=0) * 1000.0  # m → mm
                    precip_flat  = np.clip(
                        precip_daily.flatten()[in_algeria], 0.0, None
                    )
                else:
                    precip_flat = np.zeros(len(cell_ids))

                # NOON SNAPSHOT: 11:00 UTC (12:00 Algeria local)
                hour_mask = day_times.hour == 11
                if hour_mask.sum() == 0:
                    hour_mask = day_times.hour == 12
                if hour_mask.sum() == 0:
                    self.logger.warning(
                        f"  No solar-noon data for {date_str} — skipping"
                    )
                    continue

                noon_idx  = np.where(hour_mask)[0][0]
                noon_ds   = day_ds.isel(time=noon_idx)

                def extract(var):
                    if var in ds:
                        return noon_ds[var].values.flatten()[in_algeria]
                    return np.full(len(cell_ids), np.nan)

                temp_k     = extract("t2m")
                dewpoint_k = extract("d2m")
                u10        = extract("u10")
                v10        = extract("v10")
                soil_moist = extract("swvl1")

                temp_c         = temp_k - 273.15
                rh             = compute_relative_humidity(temp_k, dewpoint_k)
                wind_speed_kmh = compute_wind_speed_kmh(u10, v10)
                wind_dir       = compute_wind_direction(u10, v10)

                day_df = pd.DataFrame({
                    "date"          : date_str,
                    "era5_cell_id"  : cell_ids,
                    "latitude"      : cell_lats,
                    "longitude"     : cell_lons,
                    "temp_c"        : temp_c.round(2),
                    "rh"            : rh.round(2),
                    "wind_speed_kmh": wind_speed_kmh.round(2),
                    "wind_dir"      : wind_dir.round(1),
                    "precip_mm"     : precip_flat.round(2),
                    "soil_moisture" : soil_moist.round(4),
                    "month"         : date.month,
                })

                all_months.append(day_df)

            ds.close()

        df = pd.concat(all_months, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])

        #  DEDUP (safeguard against overlapping ingest runs) 
        before = len(df)
        df = df.drop_duplicates(subset=["date", "era5_cell_id"], keep="last")
        if len(df) < before:
            self.logger.warning(
                f"Dropped {before - len(df)} duplicate (date, era5_cell_id) rows"
            )

        self.logger.info(
            f"Combined: {df.shape} | "
            f"cells: {df['era5_cell_id'].nunique()} | "
            f"dates: {df['date'].nunique()}"
        )

        #  FILL NaN BORDER CELLS before FWI 
        # 11 edge cells have no noon data on any date — fill from spatial
        # median of their date to avoid propagating NaN through FWI computation.
        met_cols = ["temp_c", "rh", "wind_speed_kmh", "wind_dir",
                    "precip_mm", "soil_moisture"]
        nan_counts = df[met_cols].isna().sum()
        if nan_counts.any():
            self.logger.warning(
                f"NaN met values before FWI — filling with daily spatial median:\n"
                f"{nan_counts[nan_counts > 0].to_dict()}"
            )
            for col in met_cols:
                if df[col].isna().any():
                    daily_median = df.groupby("date")[col].transform("median")
                    df[col] = df[col].fillna(daily_median)

            # If still NaN (entire date missing), fill with column median
            for col in met_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
                    self.logger.warning(
                        f"  {col}: still NaN after daily fill — "
                        f"used global median as fallback"
                    )

        # FWI (must run before season filter — codes have memory) 
        self.logger.info("Computing FWI components (vectorized)...")
        df = compute_fwi_series(df)

        # FWI SANITY LOG 
        n_extreme = (df["FWI"] > 200).sum()
        if n_extreme > 0:
            self.logger.warning(
                f"{n_extreme} rows with FWI > 200 (extreme but physically possible) — "
                f"sample: {df[df['FWI']>200][['date','era5_cell_id','FWI']].head(3).to_dict('records')}"
            )

        self.logger.info(f"Final shape: {df.shape}")
        self.logger.info(f"Date range:  {df['date'].min().date()} -> {df['date'].max().date()}")
        self.logger.info(f"FWI range:   {df['FWI'].min():.1f} -> {df['FWI'].max():.1f}")

        fire_months_cfg = self.config.get("training", {}).get(
            "fire_season_months", [7, 8, 9]
        )
        fire_season = df[df["month"].isin(fire_months_cfg)]
        off_season  = df[~df["month"].isin(fire_months_cfg)]
        off_season_avg = (
            f"{off_season['FWI'].mean():.1f}" if len(off_season) > 0 else "n/a"
        )
        self.logger.info(
            f"Fire season avg FWI: {fire_season['FWI'].mean():.1f} "
            f"(vs off-season: {off_season_avg})"
        )

        # FIRE SEASON FILTER (after FWI) 
        df = filter_fire_season(df, self.config, date_col="date")

        start_str = df["date"].min().strftime("%Y%m%d")
        end_str   = df["date"].max().strftime("%Y%m%d")
        out_path  = out_dir / f"era5_{start_str}_{end_str}.parquet"
        df.to_parquet(out_path, index=False)
        self.logger.info(f"Saved → {out_path}")

    def load(self):
        """Return most recent curated ERA5 parquet."""
        out_dir = Path(self.config["era5"]["paths"]["curated"])
        files = sorted(out_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError("Run curate() first")
        df = pd.read_parquet(files[-1])
        self.logger.info(f"Loaded ERA5: {df.shape}")
        return df

# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["era5"]["paths"]
    source = ERA5Source(config)

    source.ingest(start_date="2015-06-01", end_date="2020-12-30")
    source.curate()

    df = source.load()
    print(f"\nERA5 complete")
    print(f"   Shape:      {df.shape}")
    print(f"   Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"   ERA5 cells: {df['era5_cell_id'].nunique()}")
    print(f"   Columns:    {list(df.columns)}")
    print(df)