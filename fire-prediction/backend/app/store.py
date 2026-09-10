"""
File-backed prediction store for the Algeria Wildfire Risk API.

Expected artifacts under settings.serving_dir:
  predictions_latest.json
  predictions_latest.geojson
  predictions_YYYY-MM-DD.json
  run_status.json
"""

import json
import re
from pathlib import Path

from .settings import settings


EMPTY_GEOJSON = {
    "type": "FeatureCollection",
    "features": [],
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_date(payload, fallback: str | None = None) -> str | None:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first.get("prediction_date") or first.get("target_date")
    if isinstance(payload, dict):
        direct_date = payload.get("prediction_date") or payload.get("target_date")
        if direct_date:
            return direct_date
        if payload.get("type") == "FeatureCollection":
            features = payload.get("features") or []
            if features and isinstance(features[0], dict):
                properties = features[0].get("properties") or {}
                return properties.get("prediction_date") or properties.get("target_date")
    return fallback


def _artifact_sort_key(path: Path, payload) -> tuple[str, float]:
    match = re.search(r"predictions_(\d{4}-\d{2}-\d{2})", path.name)
    date_value = _prediction_date(payload, match.group(1) if match else None) or ""
    return date_value, path.stat().st_mtime


def _valid_json_artifacts() -> list[tuple[Path, object]]:
    artifacts = []
    for path in settings.serving_dir.glob("predictions_*.json"):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            artifacts.append((path, payload))
    return artifacts


def _latest_json() -> list[dict]:
    artifacts = _valid_json_artifacts()
    if not artifacts:
        return []
    _, payload = max(artifacts, key=lambda item: _artifact_sort_key(*item))
    return payload


def _latest_geojson() -> dict:
    artifacts = []
    for path in settings.serving_dir.glob("predictions_*.geojson"):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
            artifacts.append((path, payload))
        elif isinstance(payload, list):
            artifacts.append((path, {"type": "FeatureCollection", "features": payload}))
    if not artifacts:
        return EMPTY_GEOJSON.copy()
    _, payload = max(artifacts, key=lambda item: _artifact_sort_key(*item))
    return payload


def get_run_status() -> dict:
    """Read and normalize the pipeline status used by the header."""
    path = settings.serving_dir / "run_status.json"
    if not path.exists():
        return {
            "overall_status": "red",
            "last_successful_run": None,
            "model_version": None,
            "last_training_date": None,
            "message": "run_status.json was not found in data/serving",
        }

    try:
        status = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {
            "overall_status": "red",
            "last_successful_run": None,
            "model_version": None,
            "last_training_date": None,
            "message": "run_status.json could not be read",
        }

    if not isinstance(status, dict):
        return {
            "overall_status": "red",
            "last_successful_run": None,
            "model_version": None,
            "last_training_date": None,
            "message": "run_status.json did not contain an object",
        }

    status.setdefault(
        "overall_status",
        status.get("status") or status.get("pipeline_status") or status.get("state") or "unknown",
    )
    status.setdefault(
        "last_successful_run",
        status.get("last_run") or status.get("completed_at") or status.get("finished_at"),
    )
    status.setdefault("model_version", status.get("version") or status.get("modelVersion"))
    status.setdefault(
        "last_training_date",
        status.get("model_training_date")
        or status.get("training_date")
        or status.get("trained_at")
        or status.get("last_training"),
    )
    return status


def get_all_predictions() -> list[dict]:
    return _latest_json()


def get_commune_prediction(commune_id: str) -> dict | None:
    for prediction in _latest_json():
        if str(prediction.get("commune_id")) == str(commune_id):
            return prediction
    return None


def get_commune_lookup(commune_id: str) -> dict:
    prediction = get_commune_prediction(commune_id)
    if prediction:
        return prediction

    for feature in _latest_geojson().get("features", []):
        properties = feature.get("properties") or {}
        identifiers = [
            properties.get("commune_id"),
            properties.get("id"),
            properties.get("GID_3"),
            properties.get("ID_3"),
            properties.get("ADM3_PCODE"),
        ]
        if any(value is not None and str(value) == str(commune_id) for value in identifiers):
            return {
                "commune_id": str(commune_id),
                "commune_name": properties.get("commune_name") or properties.get("NAME_3") or properties.get("name"),
                "wilaya_name": properties.get("wilaya_name") or properties.get("NAME_2") or properties.get("wilaya"),
                "prediction_available": False,
                "reason_code": "not_in_latest_prediction_artifact",
                "reason": properties.get("no_prediction_reason") or (
                    "No current prediction record was written for this commune. "
                    "This is an availability state, not a LOW-risk classification. "
                    "The available serving artifacts do not contain a commune-specific "
                    "failure reason, so this cannot be described as proof that the "
                    "commune is outside a fire-prone area."
                ),
                "latest_prediction_date": (get_summary() or {}).get("prediction_date"),
            }

    return {
        "commune_id": str(commune_id),
        "prediction_available": False,
        "reason_code": "commune_not_found_in_serving_geojson",
        "reason": (
            "This commune is not present in the latest serving GeoJSON, so the "
            "current run provides no geometry or prediction record for it."
        ),
        "latest_prediction_date": (get_summary() or {}).get("prediction_date"),
    }


def get_geojson() -> dict:
    """Return the latest serving GeoJSON directly for the Leaflet map."""
    geojson = _latest_geojson()
    predictions = _latest_json()
    features = geojson.get("features", [])
    metadata = dict(geojson.get("metadata") or {})
    metadata.setdefault("all_communes", len(features))
    flagged_predictions = [
        feature for feature in features
        if (feature.get("properties") or {}).get("prediction_available") is True
    ]
    metadata.setdefault(
        "predicted_communes",
        len(flagged_predictions) if flagged_predictions else len(predictions),
    )
    metadata.setdefault(
        "unpredicted_communes",
        max(0, len(features) - metadata["predicted_communes"]),
    )
    metadata.setdefault("source", "data/serving/predictions_latest.geojson")
    metadata.setdefault("prediction_date", _prediction_date(predictions))
    if len(features) < len(predictions):
        metadata.setdefault(
            "warning",
            "predictions_latest.geojson contains fewer geometries than predictions_latest.json",
        )
    return {**geojson, "metadata": metadata}


def get_summary() -> dict:
    predictions = _latest_json()
    if not predictions:
        return {
            "HIGH": 0,
            "MODERATE": 0,
            "LOW": 0,
            "total": 0,
            "target_date": None,
            "prediction_date": None,
            "weather_staleness_days": None,
            "data_quality_status": None,
        }

    counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
    for prediction in predictions:
        label = str(prediction.get("risk_label", "LOW")).upper()
        counts[label] = counts.get(label, 0) + 1

    first = predictions[0]
    return {
        **counts,
        "total": len(predictions),
        "target_date": first.get("target_date"),
        "prediction_date": first.get("prediction_date"),
        "weather_staleness_days": first.get("weather_staleness_days"),
        "data_quality_status": first.get("data_quality_status"),
    }


def get_priority(limit: int = 20) -> list[dict]:
    predictions = _latest_json()
    elevated = [
        prediction
        for prediction in predictions
        if prediction.get("risk_class", 0) > 0
    ]
    elevated.sort(
        key=lambda prediction: prediction.get("urgency_score", 0),
        reverse=True,
    )
    return elevated[: max(1, min(limit, 100))]


def get_wilaya_stats() -> list[dict]:
    predictions = _latest_json()
    wilayas: dict[str, dict] = {}

    for prediction in predictions:
        name = prediction.get("wilaya_name", "Unknown")
        if name not in wilayas:
            wilayas[name] = {
                "wilaya_name": name,
                "HIGH": 0,
                "MODERATE": 0,
                "LOW": 0,
                "total": 0,
                "urgency_total": 0.0,
                "fire_detections": 0.0,
            }

        label = str(prediction.get("risk_label", "LOW")).upper()
        wilayas[name][label] = wilayas[name].get(label, 0) + 1
        wilayas[name]["total"] += 1
        wilayas[name]["urgency_total"] += float(prediction.get("urgency_score") or 0)
        wilayas[name]["fire_detections"] += float(prediction.get("fire_count") or 0)

    result = []
    for item in wilayas.values():
        urgency_total = item.pop("urgency_total")
        item["average_urgency"] = urgency_total / item["total"] if item["total"] else 0
        item["elevated"] = item["HIGH"] + item["MODERATE"]
        item["elevated_share"] = item["elevated"] / item["total"] if item["total"] else 0
        result.append(item)

    result.sort(key=lambda item: (item["elevated"], item["average_urgency"]), reverse=True)
    return result


def _archive_date(path: Path) -> str | None:
    match = re.search(r"predictions_(\d{4}-\d{2}-\d{2})\.json$", path.name)
    return match.group(1) if match else None


def get_historical(commune_id: str | None = None, limit: int = 30) -> list[dict]:
    """Read dated prediction archives for national or commune history."""
    archive_paths = sorted(settings.serving_dir.glob("predictions_*.json"), key=lambda path: path.name)
    rows = []

    for path in archive_paths:
        archive_date = _archive_date(path)
        if not archive_date:
            continue
        try:
            predictions = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(predictions, list):
            continue

        selected = [
            item
            for item in predictions
            if commune_id is None or str(item.get("commune_id")) == str(commune_id)
        ]
        if commune_id is not None and not selected:
            continue

        if commune_id is not None:
            item = selected[0]
            rows.append(
                {
                    "date": item.get("prediction_date") or archive_date,
                    "commune_id": commune_id,
                    "risk_label": item.get("risk_label"),
                    "prob_high": item.get("prob_high"),
                    "urgency_score": item.get("urgency_score"),
                }
            )
        else:
            counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
            for item in selected:
                label = str(item.get("risk_label", "LOW")).upper()
                counts[label] = counts.get(label, 0) + 1
            rows.append({"date": archive_date, **counts, "total": len(selected)})

    rows.sort(key=lambda item: item["date"])
    return rows[-max(1, min(limit, 180)) :]
