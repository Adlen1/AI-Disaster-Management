from typing import Optional, Any

from pydantic import BaseModel, Field


class Driver(BaseModel):
    feature: str
    shap: float
    value: float


class CommuneUnavailable(BaseModel):
    commune_id: str
    commune_name: Optional[str] = None
    wilaya_name: Optional[str] = None
    prediction_available: bool = False
    reason_code: str
    reason: str
    latest_prediction_date: Optional[str] = None


class CommunePrediction(BaseModel):
    commune_id: str
    commune_name: str
    wilaya_id: Optional[str] = None
    wilaya_name: str
    risk_class: int
    risk_label: str
    prob_low: float
    prob_moderate: float
    prob_high: float
    urgency_score: float
    urgency_label: str
    low_confidence: bool
    borderline: bool
    top_drivers: list[Driver] = Field(default_factory=list)
    fire_count: Optional[float] = 0
    mean_frp: Optional[float] = 0
    temp_c: Optional[float] = None
    rh: Optional[float] = None
    FWI: Optional[float] = None
    DC: Optional[float] = None
    DMC: Optional[float] = None
    NDVI: Optional[float] = None
    NBR: Optional[float] = None
    pop_density_mean: Optional[float] = None
    road_distance_mean_km: Optional[float] = None
    burnable_fraction: Optional[float] = None
    forest_fraction: Optional[float] = None
    weather_observation_date: Optional[str] = None
    weather_staleness_days: Optional[int] = None
    sentinel_observation_date: Optional[str] = None
    prediction_date: Optional[str] = None
    target_date: Optional[str] = None
    data_quality_status: Optional[str] = None
    model_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    xai_method: Optional[str] = None
    historical_risk: Optional[list[float]] = None


class DailySummary(BaseModel):
    HIGH: int
    MODERATE: int
    LOW: int
    total: int
    target_date: Optional[str]
    prediction_date: Optional[str]
    weather_staleness_days: Optional[int]
    data_quality_status: Optional[str]
