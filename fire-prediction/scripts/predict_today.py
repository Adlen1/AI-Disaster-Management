"""Build tomorrow's commune-level wildfire-risk predictions.

Run from the repository root after operational ingestion:

    python scripts/orchestrator.py --mode operational
    python scripts/predict_today.py

The feature date is today; the target date is tomorrow. Dynamic observations are
read from operational curated data. Frozen model-serving artifacts and static
features remain in the training/model directories.

Data source latency reference (structural, not failures):
  - FIRMS NRT VIIRS : < 3 hours  → staleness > 2 days is a real gap
  - ERA5 / ERA5T    : ~5 days    → staleness ≤ 7 days is normal operation;
                                    flag only beyond ERA5_DEGRADED_THRESH
  - Sentinel-2      : 5-day revisit + cloud cover, monthly composite
                                  → flag only beyond SENTINEL_DEGRADED_THRESH
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils.logger import setup_logger

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logger = setup_logger().bind(source="predict_today")
RISK_LABELS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

# ── Per-source staleness thresholds ──────────────────────────────────────────
# ERA5T has a published structural lag of ~5 days. Anything up to 7 days is
# normal operation. Only flag when it genuinely exceeds expected latency.
ERA5_STRUCTURAL_LAG   = 5    # days — ECMWF published latency
ERA5_DEGRADED_THRESH  = 10   # days — flag only beyond structural lag + margin

# FIRMS NRT is available within 3 hours of satellite pass. Any multi-day gap
# is a real ingest failure.
FIRMS_DEGRADED_THRESH = 2    # days

# Sentinel-2 monthly composites. 5-day revisit + cloud cover = generous window.
SENTINEL_DEGRADED_THRESH = 35  # days


# ── Debug / inspection dumps ──────────────────────────────────────────────────
DEBUG_DUMP = True   # set True to write intermediate CSVs on each run

def progress(message: str):
    """Emit pipeline progress through the standard logging system."""
    logger.info(message)


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_parquets(directory: Path, prefix: str | None = None) -> pd.DataFrame:
    pattern = f"{prefix}_*.parquet" if prefix else "*.parquet"
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in {directory}")
    progress(f"Loading {len(files)} parquet file(s) from {directory} matching {pattern}")
    frames = []
    for path in files:
        progress(f"  reading {path.name}")
        frames.append(pd.read_parquet(path))
    result = pd.concat(frames, ignore_index=True).drop_duplicates()
    progress(f"  loaded {len(result):,} rows")
    return result


def load_firms(directory: Path) -> pd.DataFrame:
    path = directory / "firms_curated.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing operational FIRMS file: {path}")
    df = pd.read_parquet(path).rename(columns={"GID_2": "commune_id", "GID_1": "wilaya_id"})
    df["commune_id"] = df["commune_id"].astype(str)
    df["date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["date", "commune_id"])


def load_weather(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No ERA5 parquet files found in {directory}")

    def end_date(path: Path):
        match = re.search(r"_(\d{8})\.parquet$", path.name)
        return pd.to_datetime(match.group(1), format="%Y%m%d") if match else pd.Timestamp.min

    selected = max(files, key=end_date)
    progress(f"Loading authoritative ERA5 parquet: {selected.name}")
    df = pd.read_parquet(selected)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "era5_cell_id"]).drop_duplicates(
        ["era5_cell_id", "date"], keep="last"
    )
    progress(f"  loaded {len(df):,} rows from latest ERA5 coverage")
    return df


def load_sentinel(directory: Path, feature_date: pd.Timestamp) -> pd.DataFrame:
    df = load_parquets(directory, "sentinel")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["commune_id"] = df["commune_id"].astype(str)
    df = df.dropna(subset=["date", "commune_id"])
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    return df[df["date"] <= feature_date].sort_values("date").drop_duplicates(
        ["commune_id", "year", "month"], keep="last"
    )


def load_model(model_dir: Path, meta: dict):
    if meta.get("winner_model") == "CatBoost Tuned":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_dir / "winner_tuned.cbm"))
        return model
    path = model_dir / "winner_tuned.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return joblib.load(path)


def decision_rule(proba: np.ndarray, moderate_t, high_t) -> np.ndarray:
    if moderate_t is None or high_t is None:
        return np.argmax(proba, axis=1)
    out = np.zeros(len(proba), dtype=int)
    out[proba[:, 2] >= float(high_t)] = 2
    out[(proba[:, 1] >= float(moderate_t)) & (out == 0)] = 1
    return out


def aggregate_fire(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["commune_id", "date", "fire_count", "frp_total", "mean_frp"])
    out = df.groupby(["commune_id", "date"], as_index=False).agg(
        fire_count=("frp", "count"), frp_total=("frp", "sum")
    )
    out["mean_frp"] = out["frp_total"] / out["fire_count"].clip(lower=1)
    return out


def add_fire_features(communes, fire, feature_date, fire_months):
    ids = communes["commune_id"].astype(str).unique()
    season_start = pd.Timestamp(feature_date.year, min(fire_months), 1)
    dates = pd.date_range(season_start, feature_date, freq="D")
    dates = dates[dates.month.isin(fire_months)]
    dense = pd.MultiIndex.from_product([ids, dates], names=["commune_id", "date"]).to_frame(index=False)
    dense = dense.merge(fire, on=["commune_id", "date"], how="left")
    dense[["fire_count", "frp_total", "mean_frp"]] = dense[["fire_count", "frp_total", "mean_frp"]].fillna(0.0)
    dense = dense.merge(communes[["commune_id", "wilaya_id"]], on="commune_id", how="left")
    dense = dense.sort_values(["commune_id", "date"])
    group = dense.groupby("commune_id", sort=False)
    dense["fire_count_3d"] = group["fire_count"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    dense["fire_count_7d"] = group["fire_count"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
    dense["frp_total_7d"] = group["frp_total"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())

    def days_since(series):
        result, count = [], 0
        for value in series:
            count = 0 if value > 0 else count + 1
            result.append(count)
        return pd.Series(result, index=series.index)

    dense["days_since_fire"] = group["fire_count"].transform(days_since)
    wilaya = dense.groupby(["wilaya_id", "date"], as_index=False)["fire_count"].sum().rename(
        columns={"fire_count": "wilaya_fire_sum"}
    )
    dense = dense.merge(wilaya, on=["wilaya_id", "date"], how="left")
    dense["wilaya_fire_excl_self"] = (dense["wilaya_fire_sum"] - dense["fire_count"]).clip(lower=0)
    current = fire[fire["date"] == feature_date][
        ["commune_id", "fire_count", "frp_total", "mean_frp"]
    ].copy()
    current["fire_active"] = 1
    base = pd.DataFrame({"commune_id": ids}).merge(current, on="commune_id", how="left")
    base[["fire_count", "frp_total", "mean_frp", "fire_active"]] = base[
        ["fire_count", "frp_total", "mean_frp", "fire_active"]
    ].fillna(0.0)
    today = dense[dense["date"] == feature_date]
    return base.merge(
        today[["commune_id", "fire_count_3d", "fire_count_7d", "frp_total_7d",
               "days_since_fire", "wilaya_fire_excl_self"]],
        on="commune_id", how="left",
    )


def add_weather_features(base, communes, weather, mapping, feature_date):
    base = base.merge(mapping.drop_duplicates("commune_id"), on="commune_id", how="left")
    weather = weather[weather["date"] <= feature_date].copy()
    if weather.empty:
        raise RuntimeError("No operational ERA5 observation is available on or before the feature date")
    latest = weather.sort_values("date").groupby("era5_cell_id", as_index=False).tail(1)
    base = base.merge(
        latest.drop(columns=["date", "latitude", "longitude"], errors="ignore"),
        on="era5_cell_id", how="left",
    )
    w = weather.sort_values(["era5_cell_id", "date"]).copy()
    g = w.groupby("era5_cell_id", sort=False)
    for col, fn in [("FWI", "mean"), ("DC", "mean"), ("DMC", "mean"), ("precip_mm", "sum")]:
        w[f"{col}_7d"] = g[col].transform(lambda s: getattr(s.shift(1).rolling(7, min_periods=1), fn)())
    w["FWI_trend"] = w["FWI"] - w["FWI_7d"]
    rolling_latest = w.sort_values("date").groupby("era5_cell_id", as_index=False).tail(1)
    base = base.merge(
        rolling_latest[["era5_cell_id", "FWI_7d", "DC_7d", "DMC_7d", "precip_mm_7d", "FWI_trend"]],
        on="era5_cell_id", how="left",
    )
    base = base.rename(columns={"precip_mm_7d": "precip_7d"})
    # weather_observation_date is set once here from the actual ERA5 coverage
    # and not overwritten later. This is the single source of truth.
    base["weather_observation_date"] = latest["date"].max().strftime("%Y-%m-%d")
    return base


def add_baselines(base, engineered, static, sentinel, feature_date):
    month = feature_date.month
    base["year"], base["month"] = feature_date.year, month
    base = base.merge(static, on="commune_id", how="left")
    current_sentinel = sentinel.sort_values("date").drop_duplicates("commune_id", keep="last")
    if not current_sentinel.empty:
        base = base.merge(
            current_sentinel[["commune_id", "NDVI", "NDWI", "NBR", "date"]].rename(
                columns={"date": "sentinel_observation_date"}
            ),
            on="commune_id", how="left",
        )
    else:
        base["sentinel_observation_date"] = pd.NaT
    fire_rate = pd.read_parquet(engineered / "commune_fire_rate.parquet")
    fire_rate["commune_id"] = fire_rate["commune_id"].astype(str)
    base = base.merge(fire_rate, on="commune_id", how="left")
    base["commune_fire_rate"] = base["commune_fire_rate"].fillna(fire_rate["commune_fire_rate"].mean())
    nbr = pd.read_parquet(engineered / "nbr_baselines.parquet")
    nbr["commune_id"] = nbr["commune_id"].astype(str)
    nbr_col = "NBR_baseline" if "NBR_baseline" in nbr.columns else "_NBR_base"
    base = base.merge(nbr[["commune_id", "month", nbr_col]], on=["commune_id", "month"], how="left")
    base["NBR_anomaly"] = base["NBR"] - base[nbr_col].fillna(nbr[nbr_col].mean())
    ffmc = pd.read_parquet(engineered / "anomaly_baselines.parquet")
    ffmc["commune_id"] = ffmc["commune_id"].astype(str)
    ffmc_col = "FFMC_baseline" if "FFMC_baseline" in ffmc.columns else "_FFMC_base"
    base = base.merge(ffmc[["commune_id", "month", ffmc_col]], on=["commune_id", "month"], how="left")
    base["FFMC_anomaly"] = base["FFMC"] - base[ffmc_col].fillna(ffmc[ffmc_col].mean())
    doy = feature_date.dayofyear
    base["month_sin"] = np.sin(2 * np.pi * month / 12)
    base["month_cos"] = np.cos(2 * np.pi * month / 12)
    base["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    base["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return base


def add_urgency_and_shap(base, X, proba, preds, model_features, xai_dir, model):
    cfg_path = xai_dir / "06_urgency_config.json"
    equity_path = xai_dir / "06_equity_analysis.json"
    urgency = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    caps = urgency.get("normalization_caps", {})
    weights = urgency.get("weights", {
        "risk_proba_high": 0.50,
        "pop_density": 0.20,
        "road_distance": 0.15,
        "burnable_fraction": 0.15,
    })
    thresholds = urgency.get("thresholds", {"CRITICAL": 70, "HIGH PRIORITY": 50, "MONITOR": 30})
    pop_cap  = caps.get("pop_density_cap")  or max(base["pop_density_mean"].quantile(0.95), 1e-9)
    road_cap = caps.get("road_distance_cap") or max(base["road_distance_mean_km"].quantile(0.95), 1e-9)
    pop  = base["pop_density_mean"].fillna(0).clip(lower=0) / pop_cap
    road = base["road_distance_mean_km"].fillna(0).clip(lower=0) / road_cap
    fuel = base["burnable_fraction"].fillna(0).clip(0, 1)
    score = 100 * (
        weights.get("risk_proba_high", 0.50) * proba[:, 2]
        + weights.get("pop_density", 0.20)   * pop.clip(upper=1)
        + weights.get("road_distance", 0.15) * road.clip(upper=1)
        + weights.get("burnable_fraction", 0.15) * fuel
    )
    base["urgency_score"] = np.round(score, 1)
    base["urgency_label"] = np.select(
        [
            score >= thresholds.get("CRITICAL", 70),
            score >= thresholds.get("HIGH PRIORITY", 50),
            score >= thresholds.get("MONITOR", 30),
        ],
        ["CRITICAL", "HIGH PRIORITY", "MONITOR"],
        default="ROUTINE",
    )
    low_confidence_wilayas = []
    if equity_path.exists():
        low_confidence_wilayas = json.loads(equity_path.read_text()).get("low_confidence_wilayas", [])
    base["low_confidence"] = base["wilaya_name"].astype(str).isin(
        {str(x) for x in low_confidence_wilayas}
    )
    base["top_drivers"] = [[] for _ in range(len(base))]
    base["xai_method"] = "tree_path_dependent"

    if len(X) == 0:
        progress("No modeled communes; skipping SHAP")
        return base

    try:
        import shap as shap_lib

        progress("Creating fresh tree-path-dependent SHAP explainer")
        explainer = shap_lib.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        progress(f"Computing class-specific SHAP values for {len(X):,} modeled communes")
        raw = explainer.shap_values(X)
        if isinstance(raw, list):
            values = np.stack([np.asarray(v) for v in raw], axis=2)
        else:
            values = np.asarray(raw)

        class_count = proba.shape[1]
        expected = (len(X), len(model_features), class_count)
        if values.ndim != 3 or values.shape != expected:
            raise RuntimeError(f"Unexpected SHAP shape: {values.shape}; expected {expected}")

        for original_row, predicted_class in enumerate(preds):
            class_values = values[original_row, :, int(predicted_class)]
            order = np.argsort(np.abs(class_values))[::-1][:3]
            base.at[original_row, "top_drivers"] = [
                {
                    "feature": model_features[j],
                    "shap": round(float(class_values[j]), 4),
                    "value": float(X.iloc[original_row, j]),
                }
                for j in order
            ]
        progress("Completed SHAP explanations for every modeled commune")
    except Exception:
        logger.exception("Operational SHAP explanation generation failed")
        raise
    return base


def compute_source_staleness(
    feature_date: pd.Timestamp,
    weather: pd.DataFrame,
    firms: pd.DataFrame,
    sentinel: pd.DataFrame,
) -> dict:
    """
    Compute per-source staleness and degradation flags.

    ERA5 has a structural lag of ~5 days (ERA5T published latency).
    A weather_lag ≤ ERA5_DEGRADED_THRESH is expected normal operation —
    it does NOT indicate a problem and should NOT be surfaced as a warning.
    Only flag when staleness meaningfully exceeds the structural lag.

    FIRMS NRT is available within 3 hours of satellite overpass.
    Any gap beyond FIRMS_DEGRADED_THRESH days is a real ingest failure.

    Sentinel-2 monthly composites have a 5-day revisit + cloud cover.
    Flag only beyond SENTINEL_DEGRADED_THRESH days.
    """
    # ERA5
    weather_date = pd.to_datetime(weather["date"]).max()
    weather_lag  = int((feature_date - weather_date).days)
    era5_degraded = weather_lag > ERA5_DEGRADED_THRESH

    # FIRMS
    if not firms.empty:
        firms_date = pd.to_datetime(firms["date"], errors="coerce").max()
        firms_lag  = int((feature_date - firms_date).days) if not pd.isna(firms_date) else 999
    else:
        firms_date = pd.NaT
        firms_lag  = 999
    firms_degraded = firms_lag > FIRMS_DEGRADED_THRESH

    # Sentinel
    if not sentinel.empty:
        sentinel_date = pd.to_datetime(sentinel["date"]).max()
        sentinel_lag  = int((feature_date - sentinel_date).days) if not pd.isna(sentinel_date) else 999
    else:
        sentinel_date = pd.NaT
        sentinel_lag  = 999
    sentinel_degraded = sentinel_lag > SENTINEL_DEGRADED_THRESH

    # Log the breakdown so operators can see exactly what's happening
    progress(
        f"Source staleness — "
        f"ERA5: {weather_lag}d (structural={ERA5_STRUCTURAL_LAG}d, "
        f"thresh={ERA5_DEGRADED_THRESH}d, degraded={era5_degraded}) | "
        f"FIRMS: {firms_lag}d (thresh={FIRMS_DEGRADED_THRESH}d, degraded={firms_degraded}) | "
        f"Sentinel: {sentinel_lag}d (thresh={SENTINEL_DEGRADED_THRESH}d, degraded={sentinel_degraded})"
    )

    return {
        "era5": {
            "date": weather_date,
            "lag": weather_lag,
            "degraded": era5_degraded,
        },
        "firms": {
            "date": firms_date,
            "lag": firms_lag,
            "degraded": firms_degraded,
        },
        "sentinel": {
            "date": sentinel_date,
            "lag": sentinel_lag,
            "degraded": sentinel_degraded,
        },
    }


def build_data_quality_status(staleness: dict) -> str:
    """
    Build a data_quality_status string from per-source staleness results.

    Returns "OK" when all sources are within normal latency bounds.
    Returns a comma-separated list of specific flags when any source is degraded,
    e.g. "ERA5_STALE_12D, FIRMS_STALE_4D".

    This replaces the old binary DEGRADED_WEATHER_FRESHNESS which fired on
    every single run because ERA5's structural 5-day lag always triggered it.
    """
    flags = []
    if staleness["era5"]["degraded"]:
        flags.append(f"ERA5_STALE_{staleness['era5']['lag']}D")
    if staleness["firms"]["degraded"]:
        flags.append(f"FIRMS_STALE_{staleness['firms']['lag']}D")
    if staleness["sentinel"]["degraded"]:
        flags.append(f"SENTINEL_STALE_{staleness['sentinel']['lag']}D")
    return ", ".join(flags) if flags else "OK"


def run(feature_date: pd.Timestamp, root: Path, config_path: Path):
    # Stage 1: feature date and next-day prediction target
    target_date = feature_date + pd.Timedelta(days=1)
    progress(
        f"Starting prediction for feature date {feature_date.date()} "
        f"-> target {target_date.date()}"
    )
    progress(f"Repository root: {root}")
    progress(f"Loading configuration: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())
    engineered = resolve(root, cfg["training"]["engineered"])

    # Stage 2: frozen spatial/static artifacts
    progress("Loading frozen training artifacts and commune static features")
    static = pd.read_parquet(resolve(root, cfg["training"]["integrated"]) / "commune_static_features.parquet")
    static["commune_id"] = static["commune_id"].astype(str)
    boundaries = resolve(root, cfg["gadm"]["paths"]["curated"]) / "algeria_communes.gpkg"
    communes = gpd.read_file(boundaries).rename(columns={
        "GID_1": "wilaya_id", "NAME_1": "wilaya_name",
        "GID_2": "commune_id", "NAME_2": "commune_name",
    })
    for c in ["commune_id", "wilaya_id"]:
        communes[c] = communes[c].astype(str)
    communes = communes[communes["commune_id"].isin(set(static["commune_id"]))].copy()
    progress(f"Prepared {len(communes):,} communes for prediction")
    fire_months   = cfg["training"]["fire_season_months"]
    season_start  = pd.Timestamp(feature_date.year, min(fire_months), 1)

    # Stage 3: operational FIRMS current-season archive
    progress("Loading operational FIRMS current-season history")
    firms = load_firms(resolve(root, cfg["firms"]["paths"]["curated_operational"]))
    firms = firms[(firms["date"] >= season_start) & (firms["date"] <= feature_date)]
    progress(
        f"  FIRMS coverage: "
        f"{firms['date'].min() if not firms.empty else 'none'} -> "
        f"{firms['date'].max() if not firms.empty else 'none'} "
        f"({len(firms):,} detections)"
    )

    # Stage 4: operational ERA5/FWI history
    progress("Loading operational ERA5 current-season history")
    weather = load_weather(resolve(root, cfg["era5"]["paths"]["curated_operational"]))
    weather = weather[(weather["date"] >= season_start) & (weather["date"] <= feature_date)]
    if weather.empty:
        raise RuntimeError("No current-season operational ERA5 data is available")
    progress(
        f"  ERA5 coverage: {weather['date'].min().date()} -> "
        f"{weather['date'].max().date()} ({len(weather):,} cell-day rows)"
    )

    # Stage 5: Sentinel-2 monthly composite
    progress("Loading operational Sentinel-2 monthly data")
    sentinel_op = load_sentinel(
        resolve(root, cfg["sentinel"]["paths"]["curated_operational"]),
        feature_date,
    )
    current_month_rows = sentinel_op[
        (sentinel_op["year"] == feature_date.year)
        & (sentinel_op["month"] == feature_date.month)
    ].sort_values("date").drop_duplicates("commune_id", keep="last")

    prior_rows  = sentinel_op[sentinel_op["date"] < feature_date.replace(day=1)]
    prior_latest = prior_rows.sort_values("date").drop_duplicates("commune_id", keep="last")
    if current_month_rows.empty and prior_latest.empty:
        raise RuntimeError(
            "No operational Sentinel-2 observations are available. "
            "Run orchestrator.py --mode operational first."
        )

    missing_ids  = set(communes["commune_id"]) - set(current_month_rows["commune_id"])
    fallback_rows = prior_latest[prior_latest["commune_id"].isin(missing_ids)]
    sentinel = pd.concat([current_month_rows, fallback_rows], ignore_index=True)
    sentinel = sentinel.sort_values("date").drop_duplicates("commune_id", keep="last")
    progress(
        f"  Sentinel selected: {len(current_month_rows):,} current-month rows, "
        f"{len(fallback_rows):,} prior-month fallback rows"
    )

    # ── Per-source staleness (computed once, used for status artifact and
    #    per-prediction quality flags; no double computation later) ──────────
    staleness = compute_source_staleness(feature_date, weather, firms, sentinel)
    data_quality_status = build_data_quality_status(staleness)

    # Stage 6: frozen ERA5 commune mapping
    progress("Loading frozen commune-to-ERA5 mapping")
    mapping_path = engineered / "commune_era5_mapping.parquet"
    mapping = pd.read_parquet(mapping_path)
    mapping["commune_id"] = mapping["commune_id"].astype(str)

    progress("Building dense current-season fire calendar and rolling fire features")
    base = add_fire_features(communes, aggregate_fire(firms), feature_date, fire_months)
    base = base.merge(communes.drop(columns="geometry"), on="commune_id", how="left")

    progress("Joining current-season ERA5 values and weather rolling features")
    # weather_observation_date is set inside add_weather_features from the
    # actual ERA5 latest date. It is NOT overwritten later in this function.
    base = add_weather_features(base, communes, weather, mapping, feature_date)

    serving   = resolve(root, cfg["paths"]["serving"])

    # ── DEBUG: integrated snapshot (fire + weather, pre-baselines) ────────
    if DEBUG_DUMP:
        dump_path = serving / f"debug_integrated_{feature_date:%Y-%m-%d}.csv"
        base.to_csv(dump_path, index=False)
        progress(f"  [debug] integrated snapshot → {dump_path.name}")
    # ─────────────────────────────────────────────────────────────────────


    progress("Joining Sentinel, static features, fire-rate, and anomaly baselines")
    base = add_baselines(base, engineered, static, sentinel, feature_date)
    base["frp_total"] = base["fire_count"] * base["mean_frp"]

    # ── DEBUG: fully engineered feature matrix (everything the model sees) ─
    if DEBUG_DUMP:
        dump_path = serving / f"debug_engineered_{feature_date:%Y-%m-%d}.csv"
        base.to_csv(dump_path, index=False)
        progress(f"  [debug] engineered snapshot → {dump_path.name}")
    # ─────────────────────────────────────────────────────────────────────

    # Stage 7: inference
    model_dir = resolve(root, cfg["paths"]["models"])
    xai_dir   = resolve(root, cfg["paths"]["xai"])
    # serving   = resolve(root, cfg["paths"]["serving"])
    meta      = json.loads((model_dir / "tuning_meta.json").read_text())
    features  = meta["features"]
    missing   = [f for f in features if f not in base.columns]
    if missing:
        raise RuntimeError(f"Operational feature contract missing columns: {missing}")
    progress(f"Feature contract valid: {len(features)} model features")

    X = (
        base[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    progress(f"Loading model: {meta.get('winner_model', 'unknown')}")
    model = load_model(model_dir, meta)
    progress(f"Running inference for {len(X):,} communes")
    proba = model.predict_proba(X)
    preds = decision_rule(proba, meta.get("calibration_mod_thresh"), meta.get("calibration_high_thresh"))
    progress(
        f"Predictions: LOW={(preds == 0).sum():,}, "
        f"MODERATE={(preds == 1).sum():,}, "
        f"HIGH={(preds == 2).sum():,}"
    )

    # Stage 8: urgency scores and SHAP explanations
    progress("Computing urgency scores and SHAP drivers for every modeled commune")
    base = add_urgency_and_shap(base, X, proba, preds, features, xai_dir, model)

    # ── DEBUG: post-inference snapshot (predictions + SHAP + urgency) ─────
    if DEBUG_DUMP:
        dump_path = serving / f"debug_predictions_{feature_date:%Y-%m-%d}.csv"
        # top_drivers is a list column — serialize it for CSV
        debug_df = base.copy()
        debug_df["top_drivers"] = debug_df["top_drivers"].apply(json.dumps)
        debug_df.to_csv(dump_path, index=False)
        progress(f"  [debug] predictions snapshot → {dump_path.name}")
    # ─────────────────────────────────────────────────────────────────────
    
    # Stage 9: annotate predictions
    base["prediction_date"]       = feature_date.strftime("%Y-%m-%d")
    base["target_date"]           = target_date.strftime("%Y-%m-%d")
    base["risk_class"]            = preds.astype(int)
    base["risk_label"]            = [RISK_LABELS[int(x)] for x in preds]
    # Assign proba columns first — borderline depends on prob_high existing.
    base["prob_low"]              = proba[:, 0]
    base["prob_moderate"]         = proba[:, 1]
    base["prob_high"]             = proba[:, 2]

    # ── borderline: derived from the calibrated decision threshold in meta,
    #    not hard-coded magic numbers. If you retune calibration_high_thresh,
    #    the borderline band follows automatically. ──────────────────────────
    high_t  = float(meta.get("calibration_high_thresh", 0.39))
    margin  = float(cfg.get("operational", {}).get("borderline_margin", 0.04))
    base["borderline"] = (
        (base["prob_high"] >= high_t - margin) & (base["prob_high"] < high_t + margin)
    )
    base["model_version"]         = meta.get("model_version", meta.get("winner_model", "unknown"))
    base["feature_schema_version"] = meta.get("feature_schema_version", "nb03-v1")
    # weather_observation_date already set in add_weather_features — do not overwrite
    base["weather_staleness_days"]   = staleness["era5"]["lag"]
    base["sentinel_observation_date"] = (
        staleness["sentinel"]["date"].strftime("%Y-%m-%d")
        if not pd.isna(staleness["sentinel"]["date"])
        else None
    )
    base["sentinel_staleness_days"] = staleness["sentinel"]["lag"]
    # data_quality_status: specific flags, empty ("OK") when all sources healthy.
    # ERA5 structural lag alone does not produce a flag.
    base["data_quality_status"] = data_quality_status

    # Stage 10: write outputs
    serving.mkdir(parents=True, exist_ok=True)
    cols = [
        "commune_id", "commune_name", "wilaya_id", "wilaya_name",
        "prediction_date", "target_date",
        "risk_class", "risk_label", "prob_low", "prob_moderate", "prob_high",
        "urgency_score", "urgency_label",
        "low_confidence", "borderline",
        "top_drivers", "xai_method",
        "fire_count", "mean_frp",
        "temp_c", "rh", "FWI", "DC", "DMC",
        "NDVI", "NBR",
        "pop_density_mean", "road_distance_mean_km",
        "forest_fraction", "burnable_fraction",
        "weather_observation_date", "weather_staleness_days",
        "sentinel_observation_date", "sentinel_staleness_days",
        "model_version", "feature_schema_version",
        "data_quality_status",
    ]
    result = base[[c for c in cols if c in base.columns]].copy()

    day_json   = serving / f"predictions_{feature_date:%Y-%m-%d}.json"
    latest_json = serving / "predictions_latest.json"
    day_geo    = serving / f"predictions_{feature_date:%Y-%m-%d}.geojson"
    latest_geo = serving / "predictions_latest.geojson"

    progress("Writing website JSON output")
    result.to_json(day_json,    orient="records", indent=2)
    result.to_json(latest_json, orient="records", indent=2)

    progress("Writing website GeoJSON map output")
    gdf = communes[["commune_id", "geometry"]].merge(result, on="commune_id", how="inner")
    gdf.to_file(day_geo,    driver="GeoJSON")
    gdf.to_file(latest_geo, driver="GeoJSON")

    # Stage 11: status artifact — written last so FastAPI and the frontend
    # can use its presence to confirm a fully successful run.
    def fmt_date(value):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    # overall_status is "success" when all sources are within normal latency.
    # ERA5's structural 5-day lag does NOT count as a warning.
    any_degraded = any(staleness[s]["degraded"] for s in ("era5", "firms", "sentinel"))
    overall_status = "degraded" if any_degraded else "success"

    # weather_staleness_days in the status artifact: exposed for the frontend
    # status banner, but only non-zero when ERA5 is genuinely degraded (i.e.
    # beyond structural lag). This prevents the "Data delayed" banner from
    # showing on every normal run.
    status = {
        "overall_status": overall_status,
        "last_successful_run": datetime.now(timezone.utc).isoformat(),
        "prediction_date": feature_date.strftime("%Y-%m-%d"),
        "target_date":     target_date.strftime("%Y-%m-%d"),
        "commune_count":   int(len(result)),
        "model_version":   str(meta.get("model_version", meta.get("winner_model", "unknown"))),
        "feature_schema_version": str(meta.get("feature_schema_version", "nb03-v1")),
        "xai_method":      "tree_path_dependent",
        "sources": {
            "firms": {
                "status":                "degraded" if staleness["firms"]["degraded"] else "ok",
                "last_observation_date": fmt_date(staleness["firms"]["date"]),
                "staleness_days":        staleness["firms"]["lag"],
            },
            "era5": {
                "status":                "degraded" if staleness["era5"]["degraded"] else "ok",
                "last_observation_date": fmt_date(staleness["era5"]["date"]),
                "staleness_days":        staleness["era5"]["lag"],
                "structural_lag_days":   ERA5_STRUCTURAL_LAG,
                "note": (
                    f"ERA5T has a {ERA5_STRUCTURAL_LAG}-day structural lag from ECMWF. "
                    f"Values up to {ERA5_DEGRADED_THRESH} days are normal operation."
                ),
            },
            "sentinel": {
                "status":                "degraded" if staleness["sentinel"]["degraded"] else "ok",
                "last_observation_date": fmt_date(staleness["sentinel"]["date"]),
                "staleness_days":        staleness["sentinel"]["lag"],
            },
        },
        # Exposed for the frontend status banner.
        # 0 when ERA5 is within normal latency; actual lag only when degraded.
        "weather_staleness_days": staleness["era5"]["lag"] if staleness["era5"]["degraded"] else 0,
        "data_quality_status":    data_quality_status,
        "error": None,
    }
    status_path = serving / "run_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    progress(f"Saved {len(result):,} commune predictions")
    logger.info(
        "ERA5 observation: {} ({} days, structural lag {}d, degraded: {})",
        fmt_date(staleness["era5"]["date"]),
        staleness["era5"]["lag"],
        ERA5_STRUCTURAL_LAG,
        staleness["era5"]["degraded"],
    )
    progress(f"Outputs: {day_json}, {latest_json}, {day_geo}, {latest_geo}")
    return day_json, day_geo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",   help="Feature date YYYY-MM-DD; defaults to today")
    parser.add_argument("--root",   default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    root        = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    config      = Path(args.config).resolve() if args.config else root / "configs" / "config.yaml"
    feature_date = pd.Timestamp(args.date).normalize() if args.date else pd.Timestamp(date.today())
    run(feature_date, root, config)