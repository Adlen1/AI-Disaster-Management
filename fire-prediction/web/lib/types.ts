export type Driver = { feature: string; shap: number; value: number };

export type CommunePrediction = {
  commune_id: string; commune_name: string; wilaya_id?: string | null; wilaya_name: string;
  risk_class: number; risk_label: string; prob_low: number; prob_moderate: number; prob_high: number;
  urgency_score: number; urgency_label: string; low_confidence: boolean; borderline: boolean;
  top_drivers: Driver[]; prediction_date?: string | null; target_date?: string | null;
  model_version?: string | null; historical_risk?: number[] | null; fire_count?: number | null;
  mean_frp?: number | null; temp_c?: number | null; rh?: number | null; FWI?: number | null;
  DC?: number | null; DMC?: number | null; NDVI?: number | null; NBR?: number | null;
  pop_density_mean?: number | null; road_distance_mean_km?: number | null;
  burnable_fraction?: number | null; forest_fraction?: number | null;
  weather_observation_date?: string | null; weather_staleness_days?: number | null;
  sentinel_observation_date?: string | null; data_quality_status?: string | null;
  feature_schema_version?: string | null; xai_method?: string | null;
};

export type CommuneUnavailable = {
  commune_id: string; commune_name?: string | null; wilaya_name?: string | null;
  prediction_available: false; reason_code: string; reason: string; latest_prediction_date?: string | null;
};

export type CommuneResponse = CommunePrediction | CommuneUnavailable;
export type DailySummary = { HIGH: number; MODERATE: number; LOW: number; total: number; target_date?: string | null; prediction_date?: string | null; weather_staleness_days?: number | null; data_quality_status?: string | null };
export type WilayaStats = { wilaya_name: string; HIGH: number; MODERATE: number; LOW: number; total: number; elevated?: number; elevated_share?: number; average_urgency?: number; fire_detections?: number };
export type HistoricalNationalPoint = { date: string; HIGH: number; MODERATE: number; LOW: number; total: number };
export type HistoricalCommunePoint = { date: string; commune_id: string; risk_label?: string | null; prob_high?: number | null; urgency_score?: number | null };
export type RunStatus = {
  last_successful_run?: string | null;
  overall_status?: string | null;
  model_version?: string | null;
  last_training_date?: string | null;
  message?: string | null;
  prediction_date?: string | null;
  target_date?: string | null;
  commune_count?: number | null;
  weather_staleness_days?: number | null;
  data_quality_status?: string | null;

  sources?: {
    firms?: {
      status?: "ok" | "degraded";
      last_observation_date?: string | null;
      staleness_days?: number;
    };
    era5?: {
      status?: "ok" | "degraded";
      last_observation_date?: string | null;
      staleness_days?: number;
      structural_lag_days?: number;
      note?: string;
    };
    sentinel?: {
      status?: "ok" | "degraded";
      last_observation_date?: string | null;
      staleness_days?: number;
    };
  };

  [key: string]: unknown;
};export type GeoJsonFeature = { properties?: Record<string, unknown>; geometry?: { type: string; coordinates: unknown } | null };
export type GeoJsonResponse = { type: "FeatureCollection"; features: GeoJsonFeature[]; metadata?: { boundary_source?: string; all_communes?: number; predicted_communes?: number; unpredicted_communes?: number; warning?: string; source?: string; prediction_date?: string | null; target_date?: string | null } };
