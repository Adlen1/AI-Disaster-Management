import type { CommunePrediction, CommuneResponse, DailySummary, GeoJsonResponse, HistoricalCommunePoint, HistoricalNationalPoint, RunStatus, WilayaStats } from "./types";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "")).replace(/\/$/, "");
export function apiUrl(path: string) { return `${API_BASE_URL}${path}`; }

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try { response = await fetch(apiUrl(path), { signal, headers: { Accept: "application/json" }, cache: "no-store" }); }
  catch { throw new Error(`Could not reach the wildfire API at ${API_BASE_URL || "the current frontend origin"}. Set NEXT_PUBLIC_API_URL if the backend runs elsewhere.`); }
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json() as Promise<T>;
}

export const fetchRunStatus = (signal?: AbortSignal) => get<RunStatus>("/api/run-status", signal);
export const fetchDailyGeoJson = (signal?: AbortSignal) => get<GeoJsonResponse>("/api/predict/daily", signal);
export const fetchDailySummary = (signal?: AbortSignal) => get<DailySummary>("/api/predict/summary", signal);
export const fetchPriority = (limit = 50, signal?: AbortSignal) => get<CommunePrediction[]>(`/api/predict/priority?limit=${limit}`, signal);
export const fetchWilayaStats = (signal?: AbortSignal) => get<WilayaStats[]>("/api/predict/wilayas", signal);
export const fetchHistorical = (communeId?: string | null, limit = 30, signal?: AbortSignal) => get<HistoricalNationalPoint[] | HistoricalCommunePoint[]>(`/api/predict/history?limit=${limit}${communeId ? `&commune_id=${encodeURIComponent(communeId)}` : ""}`, signal);
export const fetchCommune = (communeId: string, signal?: AbortSignal) => get<CommuneResponse>(`/api/predict/${encodeURIComponent(communeId)}`, signal);
export function formatTimestamp(value?: string | null) { if (!value) return "Unavailable"; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeStyle: "short" }).format(date); }
