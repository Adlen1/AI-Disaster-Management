# scripts/ingest/era5.py
"""
ERA5 — Meteorological Reanalysis (Dynamic)

Key architectural decision:
  ERA5 native resolution is ~31km.
  We do NOT interpolate to the 1km terrain grid here.
  Instead we keep ERA5 at its native grid (~31km cells over Algeria)
  and store (date, era5_cell_id, lat, lon, weather_vars).

Output:
  raw/era5/era5_YYYYMM.nc           <- one NetCDF per month
  curated/era5/era5_YYYYMMDD_YYYYMMDD.parquet
  -> columns: date, era5_cell_id, lat, lon,
              temp_c, rh, wind_speed_kmh, wind_dir,
              precip_mm, soil_moisture,
              FFMC, DMC, DC, ISI, BUI, FWI
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



# ── FWI COMPUTATION ───────────────────────────────────────────────────────────
# Canadian Fire Weather Index — Van Wagner (1987)
# All formulas verified against the original paper.

def compute_relative_humidity(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    """Derive RH (%) from temperature and dewpoint (Kelvin)."""
    temp_c = temp_k - 273.15
    dew_c  = dewpoint_k - 273.15
    rh = 100 * (
        np.exp((17.625 * dew_c)  / (243.04 + dew_c)) /
        np.exp((17.625 * temp_c) / (243.04 + temp_c))
    )
    return np.clip(rh, 0, 100)


def compute_wind_speed_kmh(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Wind speed in km/h from ERA5 U/V components (m/s). FWI needs km/h."""
    return np.sqrt(u**2 + v**2) * 3.6


def compute_wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Wind direction in degrees (meteorological convention, 0=North)."""
    return (270 - np.degrees(np.arctan2(v, u))) % 360


def _ffmc_scalar(temp, rh, wind_kmh, rain, prev_ffmc):
    """
    Fine Fuel Moisture Code — Van Wagner (1987).
    Verified against authoritative pyfwi / FWI_CMIP6 implementations.
    """
    # Clamp inputs
    rh       = max(0.0, min(100.0, rh))
    wind_kmh = max(0.0, wind_kmh)
    rain     = max(0.0, rain)

    mo = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)

    if rain > 0.5:
        rf = rain - 0.5
        mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        if mo > 150:
            mr += 0.0015 * (mo - 150.0)**2 * rf**0.5
        mo = min(mr, 250.0)

    ed = (0.942 * rh**0.679
          + 11.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))

    ew = (0.618 * rh**0.753
          + 10.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))

    if mo > ed:
        ko = (0.424 * (1.0 - (rh / 100.0)**1.7)
              + 0.0694 * wind_kmh**0.5 * (1.0 - (rh / 100.0)**8))
        kd = ko * 0.581 * np.exp(0.0365 * temp)
        mo = ed + (mo - ed) * 10.0**(-kd)
    elif mo < ew:
        kl = (0.424 * (1.0 - ((100.0 - rh) / 100.0)**1.7)
              + 0.0694 * wind_kmh**0.5 * (1.0 - ((100.0 - rh) / 100.0)**8))
        kw = kl * 0.581 * np.exp(0.0365 * temp)
        mo = ew - (ew - mo) * 10.0**(-kw)

    return 59.5 * (250.0 - mo) / (147.2 + mo)


def _dmc_scalar(temp, rh, rain, month, prev_dmc):
    """
    Duff Moisture Code — Van Wagner (1987).
    Day-length factors for Northern Algeria (~28-37°N), using 'bins' method.
    """
    rh   = max(0.0, min(100.0, rh))
    rain = max(0.0, rain)

    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20.0 + np.exp(5.6348 - prev_dmc / 43.43)
        if prev_dmc <= 33:
            b = 100.0 / (0.5 + 0.3 * prev_dmc)
        elif prev_dmc <= 65:
            b = 14.0 - 1.3 * np.log(prev_dmc)
        else:
            b = 6.2 * np.log(prev_dmc) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * np.log(mr - 20.0)
        prev_dmc = max(pr, 0.0)

    if temp <= -1.1:
        return prev_dmc

    # Day-length factor for Algeria latitudes (20-33°N range → DayLength20N)
    # Algeria fire-prone north: 33-37°N → DayLength46N
    # Using the standard Canadian values (original method, appropriate for Algeria)
    Le = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0][month - 1]
    k  = 1.894 * (temp + 1.1) * (100.0 - rh) * Le * 1e-6
    return prev_dmc + 100.0 * k


def _dc_scalar(temp, rain, month, prev_dc):
    """
    Drought Code — Van Wagner (1987).
    Drying factors for Northern Algeria (North of equator, original values).
    """
    rain = max(0.0, rain)

    if rain > 2.8:
        rd     = 0.83 * rain - 1.27
        qo     = 800.0 * np.exp(-prev_dc / 400.0)
        qr     = qo + 3.937 * rd
        dr     = 400.0 * np.log(800.0 / qr)
        prev_dc = max(dr, 0.0)

    # Drying factors — original Van Wagner values (Northern hemisphere)
    Lf = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6][month - 1]

    if temp <= -2.8:
        v = Lf
    else:
        v = 0.36 * (temp + 2.8) + Lf

    v = max(v, 0.0)
    return prev_dc + 0.5 * v


def _isi_scalar(wind_kmh, ffmc_val):
    """
    Initial Spread Index — Van Wagner (1987).
    Verified: coefficient is 91.9 (not 19.115), divisor is 49,300,000.
    Edge case: if FFMC > 101, set fm = 0 (approximation artifact in FFMC formula).
    """
    wind_kmh = max(0.0, wind_kmh)

    m = 147.2 * (101.0 - ffmc_val) / (59.5 + ffmc_val)
    m = max(m, 0.0)  # handles FFMC > 101 edge case

    f_wind = np.exp(0.05039 * wind_kmh)
    f_fuel = 91.9 * np.exp(-0.1386 * m) * (1.0 + m**5.31 / 49300000.0)
    return 0.208 * f_wind * f_fuel


def _bui_scalar(dmc_val, dc_val):
    """
    Buildup Index — Van Wagner (1987).
    Handles DMC=DC=0 edge case explicitly.
    """
    dmc_val = max(dmc_val, 0.0)
    dc_val  = max(dc_val,  0.0)

    if dmc_val == 0.0 and dc_val == 0.0:
        return 0.0

    if dmc_val <= 0.4 * dc_val:
        denom = dmc_val + 0.4 * dc_val
        bui   = 0.8 * dmc_val * dc_val / denom if denom > 0 else 0.0
    else:
        bui = dmc_val - (1.0 - 0.8 * dc_val / (dmc_val + 0.4 * dc_val)) * \
              (0.92 + (0.0114 * dmc_val)**1.7)

    return max(bui, 0.0)


def _fwi_scalar(isi_val, bui_val):
    """
    Fire Weather Index — Van Wagner (1987).
    """
    isi_val = max(isi_val, 0.0)
    bui_val = max(bui_val, 0.0)

    if bui_val <= 80.0:
        fd = 0.626 * bui_val**0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui_val))

    b = 0.1 * isi_val * fd

    if b <= 0.0:
        return 0.0
    elif b > 1.0:
        return np.exp(2.72 * (0.434 * np.log(b))**0.647)
    else:
        return b


def compute_fwi_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FWI components for all ERA5 cells over time.
    Loops over TIME (unavoidable - day-to-day carry-over state).
    Loops over cells within each day (scalar formulas) - fine at this scale
    (~120 cells x ~thousands of days), not true numpy vectorization.

    Input:  df with columns [date, era5_cell_id, temp_c, rh,
                              wind_speed_kmh, precip_mm, month]
    Output: df with added columns [FFMC, DMC, DC, ISI, BUI, FWI]
    """
    df = df.copy().sort_values(["date", "era5_cell_id"])

    # Guard against duplicate (date, cell) rows - would silently break .loc[cid] below
    dupes = df.duplicated(["date", "era5_cell_id"]).sum()
    if dupes > 0:
        raise ValueError(
            f"Found {dupes} duplicate (date, era5_cell_id) rows before FWI computation. "
            f"Fix the upstream curation step - FWI carry-over state assumes exactly "
            f"one row per cell per day."
        )

    dates = sorted(df["date"].unique())
    cell_ids = sorted(df["era5_cell_id"].unique())
    n_cells = len(cell_ids)
    cell_idx = {cid: i for i, cid in enumerate(cell_ids)}

    prev_ffmc = np.full(n_cells, 85.0)
    prev_dmc = np.full(n_cells, 6.0)
    prev_dc = np.full(n_cells, 15.0)

    fwi_rows = []

    for date in dates:
        day = df[df["date"] == date].copy().set_index("era5_cell_id")

        ffmc_out = np.full(n_cells, np.nan)
        dmc_out = np.full(n_cells, np.nan)
        dc_out = np.full(n_cells, np.nan)
        isi_out = np.full(n_cells, np.nan)
        bui_out = np.full(n_cells, np.nan)
        fwi_out = np.full(n_cells, np.nan)

        month = int(pd.to_datetime(date).month)

        for cid in cell_ids:
            i = cell_idx[cid]
            if cid not in day.index:
                continue  # missing data for this cell/day -> leave NaN, carry state unchanged

            row = day.loc[cid]
            f = _ffmc_scalar(row.temp_c, row.rh, row.wind_speed_kmh, row.precip_mm, prev_ffmc[i])
            d = _dmc_scalar(row.temp_c, row.rh, row.precip_mm, month, prev_dmc[i])
            dcv = _dc_scalar(row.temp_c, row.precip_mm, month, prev_dc[i])
            isiv = _isi_scalar(row.wind_speed_kmh, f)
            buiv = _bui_scalar(d, dcv)
            fwiv = _fwi_scalar(isiv, buiv)

            ffmc_out[i], dmc_out[i], dc_out[i] = f, d, dcv
            isi_out[i], bui_out[i], fwi_out[i] = isiv, buiv, fwiv

            prev_ffmc[i], prev_dmc[i], prev_dc[i] = f, d, dcv

        day_result = day.reset_index().copy()
        idx_lookup = day_result["era5_cell_id"].map(cell_idx)
        day_result["FFMC"] = ffmc_out[idx_lookup]
        day_result["DMC"] = dmc_out[idx_lookup]
        day_result["DC"] = dc_out[idx_lookup]
        day_result["ISI"] = isi_out[idx_lookup]
        day_result["BUI"] = bui_out[idx_lookup]
        day_result["FWI"] = fwi_out[idx_lookup]
        fwi_rows.append(day_result)

    return pd.concat(fwi_rows, ignore_index=True)


# ── ERA5 SOURCE ───────────────────────────────────────────────────────────────

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
        Download ERA5 monthly NetCDF files for Algeria.
        One file per month - skips already-downloaded months.

        Precipitation note: ERA5 'total_precipitation' is an hourly
        accumulation. We download all 24 hourly steps and sum them in
        curate() to get the true 24h rainfall total for FWI computation.
        """
        if not start_date:
            start_date = datetime.today().strftime("%Y-%m-%d")
        if not end_date:
            end_date = start_date

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        bbox = self.config["algeria"]["bbox"]  # [west, south, east, north]

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
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": self.CDS_VARIABLES,
                        "year": year_str,
                        "month": month_str,
                        "day": days,
                        "time": all_hours,
                        "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
                        "data_format": "netcdf",       # new CDS-Beta key
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
        """CDS sometimes zips the response regardless of 'unarchived'. Unwrap
        without depending on dask (xr.merge instead of open_mfdataset)."""
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
                self.logger.info(f"  Merging {len(nc_inside)} files (no dask needed): "
                                  f"{[f.name for f in nc_inside]}")
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
        """The new CDS backend has two quirks this normalizes:
        1. It sometimes names the time coordinate 'valid_time' instead of
           'time' — happens especially when instant/accum variable groups
           get merged (each group can use a different name).
        2. It sometimes adds an 'expver' dimension when a request spans the
           boundary between final ERA5 and preliminary ERA5T data, giving
           each timestep two versions (expver=1 final, expver=5 preliminary,
           with NaNs filling whichever wasn't yet available). Collapse this
           by taking whichever value is not NaN, preferring expver=1.
        """
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
        """Open a single ERA5 monthly file, failing with a clear message if
        it's still a zip (e.g. an interrupted ingest left it unfixed)."""
        if zipfile.is_zipfile(nc_file):
            raise RuntimeError(
                f"{nc_file.name} is still a zip at curate() time — "
                f"the ingest()-time unwrap didn't complete. "
                f"Delete this file and re-run ingest() for this month."
            )
        ds = xr.open_dataset(nc_file, engine="netcdf4")
        ds = self._standardize_era5_dataset(ds)

        if "time" not in ds.coords:
            raise RuntimeError(
                f"{nc_file.name} has no 'time' or 'valid_time' coordinate after "
                f"standardization. Available coords: {list(ds.coords)}. "
                f"CDS may have changed its schema again — inspect the file directly."
            )
        return ds

    def curate(self):
        """
        Process ERA5 NetCDF files at native ~31km resolution.
        One row per (date, era5_cell_id) — NOT per fine 1km cell.
        Fine-grid join happens at integration time.

        Precipitation: sums all 24 hourly steps -> true daily total.
        Other variables: 11:00 UTC snapshot (~noon Algeria local time).
        """
        out_dir = Path(self.config["era5"]["paths"]["curated"])
        boundary_path = Path(self.config["gadm"]["paths"]["curated"]) / "algeria_country.gpkg"
        out_dir.mkdir(parents=True, exist_ok=True)

        nc_files = sorted(self.raw_dir.glob("era5_*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No ERA5 .nc files in {self.raw_dir}\nRun ingest() first.")

        self.logger.info(f"Processing {len(nc_files)} ERA5 monthly files")

        algeria = gpd.read_file(boundary_path).to_crs("EPSG:4326")
        algeria_geom = algeria.geometry.union_all()

        all_months = []

        for nc_file in nc_files:
            self.logger.info(f"Processing {nc_file.name}")
            ds = self._open_era5_file(nc_file)

            lats = ds.latitude.values
            lons = ds.longitude.values

            lon_grid, lat_grid = np.meshgrid(lons, lats)
            flat_lons = lon_grid.flatten()
            flat_lats = lat_grid.flatten()

            in_algeria = np.array([
                algeria_geom.contains(Point(lo, la))
                for lo, la in zip(flat_lons, flat_lats)
            ])
            cell_lons = flat_lons[in_algeria]
            cell_lats = flat_lats[in_algeria]
            cell_ids = np.where(in_algeria)[0]

            self.logger.info(f"  ERA5 cells inside Algeria: {len(cell_ids)} "
                              f"(from {len(flat_lons)} total)")

            times = pd.to_datetime(ds.time.values)
            dates = pd.DatetimeIndex(np.unique(times.date))

            for date in dates:
                date_str = str(date.date())
                date_mask = times.date == date.date()
                day_ds = ds.isel(time=date_mask)

                # Precipitation: sum all hourly steps -> daily total (mm)
                if "tp" in ds:
                    precip_daily = day_ds["tp"].values.sum(axis=0) * 1000  # m -> mm
                    precip_flat = precip_daily.flatten()[in_algeria]
                else:
                    precip_flat = np.zeros(len(cell_ids))

                # Other variables: 11:00 UTC (noon Algeria local, UTC+1)
                hour_mask = times.hour == 11
                date_hour_mask = date_mask & hour_mask
                if date_hour_mask.sum() == 0:
                    date_hour_mask = date_mask & (times.hour == 12)
                if date_hour_mask.sum() == 0:
                    self.logger.warning(f"  No noon data for {date_str} — skipping")
                    continue

                noon_ds = ds.isel(time=np.where(date_hour_mask)[0][0])

                def extract(var):
                    if var in ds:
                        return noon_ds[var].values.flatten()[in_algeria]
                    return np.full(len(cell_ids), np.nan)

                temp_k = extract("t2m")
                dewpoint_k = extract("d2m")
                u10 = extract("u10")
                v10 = extract("v10")
                soil_moist = extract("swvl1")

                temp_c = temp_k - 273.15
                rh = compute_relative_humidity(temp_k, dewpoint_k)
                wind_speed_kmh = compute_wind_speed_kmh(u10, v10)
                wind_dir = compute_wind_direction(u10, v10)

                day_df = pd.DataFrame({
                    "date": date_str,
                    "era5_cell_id": cell_ids,
                    "latitude": cell_lats,
                    "longitude": cell_lons,
                    "temp_c": temp_c.round(2),
                    "rh": rh.round(2),
                    "wind_speed_kmh": wind_speed_kmh.round(2),
                    "wind_dir": wind_dir.round(1),
                    "precip_mm": np.clip(precip_flat, 0, None).round(2),
                    "soil_moisture": soil_moist.round(4),
                    "month": date.month,
                })

                all_months.append(day_df)

            ds.close()

        df = pd.concat(all_months, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])

        self.logger.info(f"Combined: {df.shape} | cells: {df['era5_cell_id'].nunique()} | "
                          f"dates: {df['date'].nunique()}")

        self.logger.info("Computing FWI components...")
        df = compute_fwi_series(df)

        self.logger.info(f"Final shape: {df.shape}")
        self.logger.info(f"Date range: {df['date'].min()} -> {df['date'].max()}")
        self.logger.info(f"FWI range: {df['FWI'].min():.1f} -> {df['FWI'].max():.1f}")
        
        fire_months_cfg = self.config.get("training", {}).get("fire_season_months", [7, 8, 9])
        fire_season = df[df["month"].isin(fire_months_cfg)]
        off_season  = df[~df["month"].isin(fire_months_cfg)]
        
        off_season_avg = f"{off_season['FWI'].mean():.1f}" if len(off_season) > 0 else "n/a"
        self.logger.info(
            f"Fire season avg FWI: {fire_season['FWI'].mean():.1f} "
            f"(vs off-season: {off_season_avg})"
        )

        df = filter_fire_season(df, self.config, date_col="date")

        start_str = df["date"].min().strftime("%Y%m%d")
        end_str = df["date"].max().strftime("%Y%m%d")
        out_path = out_dir / f"era5_{start_str}_{end_str}.parquet"
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


if __name__ == "__main__":
    import yaml
    from utils.logger import setup_logger
    setup_logger()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"] = config["era5"]["paths"]
    source = ERA5Source(config)

    source.ingest(start_date="2015-06-01", end_date="2015-06-30")
    source.curate()

    df = source.load()
    print(f"\nERA5 complete")
    print(f"   Shape:      {df.shape}")
    print(f"   Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"   ERA5 cells: {df['era5_cell_id'].nunique()}")
    print(f"   Columns:    {list(df.columns)}")
    print(df)