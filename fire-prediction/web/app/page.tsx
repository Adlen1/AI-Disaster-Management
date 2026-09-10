"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import ActionList, { isPriority } from "../components/ActionList";
import DetailedInsights, { NationalOverview } from "../components/DetailedInsights";
import DetailPanel from "../components/DetailPanel";
import RiskMap from "../components/RiskMap";
import {
  fetchCommune, fetchDailyGeoJson, fetchDailySummary, fetchHistorical,
  fetchPriority, fetchRunStatus, fetchWilayaStats, formatTimestamp,
} from "../lib/api";
import type {
  CommunePrediction, CommuneResponse, DailySummary, GeoJsonResponse,
  HistoricalCommunePoint, HistoricalNationalPoint, RunStatus, WilayaStats,
} from "../lib/types";

type View = "simple" | "detailed";

function asPredictions(map: GeoJsonResponse | null): CommunePrediction[] {
  return (map?.features ?? [])
    .map((f) => f.properties as CommunePrediction)
    .filter((p) => p && typeof p.commune_id === "string" && typeof p.risk_label === "string");
}

function statusLabel(status: RunStatus | null, summary: DailySummary | null) {
  const stale = summary?.weather_staleness_days ?? status?.weather_staleness_days ?? 0;
  return stale > 0
    ? { text: `Weather delayed — ${stale}d`, warn: true }
    : { text: "Data current", warn: false };
}

/* ── Thin toolbar button ── */
function ToolBtn({
  onClick, children, active, label,
}: {
  onClick: () => void; children: React.ReactNode; active?: boolean; label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={[
        "px-3 py-1.5 text-xs font-semibold border transition-colors",
        active
          ? "border-brand bg-brand text-white"
          : "border-line bg-surface text-ink hover:border-brand/50 hover:text-brand",
      ].join(" ")}
      style={{ borderRadius: "var(--radius-control)" }}
    >
      {children}
    </button>
  );
}

/* ── Stat pill in the summary bar ── */
function RiskStat({
  count, label, colorClass,
}: {
  count: number; label: string; colorClass: string;
}) {
  return (
    <div className="flex items-center gap-2.5">
      <span className={`font-data text-2xl font-medium ${colorClass} leading-none`}>
        {count}
      </span>
      <span className="text-xs text-ink-muted leading-tight">{label}</span>
    </div>
  );
}

/* ── Filter row ── */
function Filters({
  query, setQuery, wilayaFilter, setWilayaFilter, riskFilter, setRiskFilter,
  priorityOnly, setPriorityOnly, clearFilters, wilayasAvailable,
}: {
  query: string; setQuery: (v: string) => void;
  wilayaFilter: string; setWilayaFilter: (v: string) => void;
  riskFilter: string; setRiskFilter: (v: string) => void;
  priorityOnly: boolean; setPriorityOnly: (v: boolean) => void;
  clearFilters: () => void;
  wilayasAvailable: string[];
}) {
  const inputCls = "px-3 py-1.5 text-xs border border-line bg-surface text-ink placeholder:text-ink-subtle focus:outline-none focus:ring-2 focus:ring-brand/20 w-full";
  const rc = "var(--radius-control)";

  return (
    <div className="flex flex-wrap gap-2 px-3 py-2.5">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search commune…"
        className={inputCls}
        style={{ borderRadius: rc, maxWidth: 200 }}
      />
      <select
        value={wilayaFilter}
        onChange={(e) => setWilayaFilter(e.target.value)}
        className={inputCls}
        style={{ borderRadius: rc, width: "auto" }}
      >
        <option value="all">All wilayas</option>
        {wilayasAvailable.map((n) => <option key={n} value={n}>{n}</option>)}
      </select>
      <select
        value={riskFilter}
        onChange={(e) => setRiskFilter(e.target.value)}
        className={inputCls}
        style={{ borderRadius: rc, width: "auto" }}
      >
        <option value="all">All risk levels</option>
        <option value="HIGH">High</option>
        <option value="MODERATE">Moderate</option>
        <option value="LOW">Low</option>
      </select>
      <ToolBtn onClick={() => setPriorityOnly(!priorityOnly)} active={priorityOnly} label="Toggle priority areas">
        {priorityOnly ? "Priority on" : "Priority areas"}
      </ToolBtn>
      <ToolBtn onClick={clearFilters} label="Reset filters">
        Reset
      </ToolBtn>
    </div>
  );
}

/* ──────────────────────────────────────────── */
function HomeContent() {
  const search = useSearchParams();
  const routeId = search.get("commune");

  const [view, setView] = useState<View>("simple");
  const [dark, setDark] = useState(false);
  const [summary, setSummary] = useState<DailySummary | null>(null);
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [map, setMap] = useState<GeoJsonResponse | null>(null);
  const [priority, setPriority] = useState<CommunePrediction[]>([]);
  const [wilayas, setWilayas] = useState<WilayaStats[]>([]);
  const [history, setHistory] = useState<HistoricalNationalPoint[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(routeId);
  const [detail, setDetail] = useState<CommuneResponse | null>(null);
  const [communeHistory, setCommuneHistory] = useState<HistoricalCommunePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [query, setQuery] = useState("");
  const [wilayaFilter, setWilayaFilter] = useState("all");
  const [riskFilter, setRiskFilter] = useState("all");
  const [priorityOnly, setPriorityOnly] = useState(false);

  /* theme */
  useEffect(() => { setDark(window.localStorage.getItem("wildfire-theme") === "dark"); }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    window.localStorage.setItem("wildfire-theme", dark ? "dark" : "light");
  }, [dark]);
  useEffect(() => { setSelectedId(routeId); }, [routeId]);

  /* initial data load */
  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    Promise.allSettled([
      fetchDailySummary(ctrl.signal),
      fetchRunStatus(ctrl.signal),
      fetchDailyGeoJson(ctrl.signal),
      fetchPriority(100, ctrl.signal),
      fetchWilayaStats(ctrl.signal),
      fetchHistorical(null, 30, ctrl.signal),
    ]).then(([a, b, c, d, e, f]) => {
      if (a.status === "fulfilled") setSummary(a.value);
      if (b.status === "fulfilled") setStatus(b.value);
      if (c.status === "fulfilled") setMap(c.value);
      if (d.status === "fulfilled") setPriority(d.value);
      if (e.status === "fulfilled") setWilayas(e.value);
      if (f.status === "fulfilled")
        setHistory(f.value.filter((item): item is HistoricalNationalPoint => "total" in item));
      const failed = [a, b, c, d, e, f].filter((r) => r.status === "rejected");
      setError(failed.length ? "Some data could not be loaded. Check that the API is running on port 8000." : null);
    }).finally(() => setLoading(false));
    return () => ctrl.abort();
  }, [refresh]);

  /* commune detail */
  useEffect(() => {
    if (!selectedId) { setDetail(null); setCommuneHistory([]); return; }
    const ctrl = new AbortController();
    fetchCommune(selectedId, ctrl.signal).then(setDetail).catch(() => setDetail(null));
    fetchHistorical(selectedId, 30, ctrl.signal)
      .then((items) => setCommuneHistory(items.filter((item): item is HistoricalCommunePoint => "commune_id" in item)))
      .catch(() => setCommuneHistory([]));
    return () => ctrl.abort();
  }, [selectedId]);

  /* derived */
  const allItems = useMemo(() => {
    const combined = [...priority, ...asPredictions(map)];
    const unique = new Map(combined.map((item) => [item.commune_id, item]));
    return [...unique.values()];
  }, [map, priority]);

  const wilayasAvailable = useMemo(
    () => [...new Set(allItems.map((item) => item.wilaya_name).filter(Boolean))].sort(),
    [allItems],
  );

  const filteredItems = useMemo(
    () =>
      allItems.filter(
        (item) =>
          (!query || `${item.commune_name} ${item.wilaya_name}`.toLowerCase().includes(query.toLowerCase())) &&
          (wilayaFilter === "all" || item.wilaya_name === wilayaFilter) &&
          (riskFilter === "all" || item.risk_label.toUpperCase() === riskFilter) &&
          (!priorityOnly || isPriority(item)),
      ),
    [allItems, query, wilayaFilter, riskFilter, priorityOnly],
  );

  const visibleIds = useMemo(() => new Set(filteredItems.map((item) => item.commune_id)), [filteredItems]);

  const selectedPrediction =
    detail && !("prediction_available" in detail)
      ? (detail as CommunePrediction)
      : allItems.find((item) => item.commune_id === selectedId) ?? null;

  const select = useCallback((id: string) => {
    setSelectedId(id);
    window.history.replaceState(null, "", `/?commune=${encodeURIComponent(id)}`);
  }, []);

  const clearFilters = () => {
    setQuery(""); setWilayaFilter("all"); setRiskFilter("all"); setPriorityOnly(false);
  };

  const targetDate = summary?.target_date ?? status?.target_date ?? "—";
  const lastUpdated = status?.last_successful_run ?? summary?.prediction_date;
  const staleDays = summary?.weather_staleness_days ?? status?.weather_staleness_days;
  const metadata = map?.metadata;
  const sl = statusLabel(status, summary);

  const filterProps = {
    query, setQuery, wilayaFilter, setWilayaFilter, riskFilter, setRiskFilter,
    priorityOnly, setPriorityOnly, clearFilters, wilayasAvailable,
  };

  const detailPanel = (
    <DetailPanel
      prediction={detail ?? selectedPrediction}
      history={communeHistory}
      loading={Boolean(selectedId && !detail)}
      onClose={() => { setSelectedId(null); setDetail(null); window.history.replaceState(null, "", "/"); }}
    />
  );

  return (
    <main className="min-h-screen bg-canvas text-ink">
      <div className="mx-auto max-w-[1600px] px-4 sm:px-8 lg:px-12">

        {/* ─── Header ─── */}
        <header className="flex items-center justify-between gap-4 border-b border-line py-4">
          <div className="flex items-center gap-4 min-w-0">
            <div>
              <p className="text-[10px] text-ink-subtle tracking-widest font-medium">Wildfire Risk DZ</p>
              <h1 className="font-data text-xl font-medium text-ink leading-tight">
                {view === "simple" ? "Next-day fire risk Simple View" : "Next-day fire risk Detailed View"}
              </h1>
            </div>
            <div className="hidden sm:block w-px h-8 bg-line" />
            <p className="hidden sm:block text-xs text-ink-muted">
              Predictions for{" "}
              <span className="font-semibold text-ink">
                {targetDate === "—" ? "next operational day" : targetDate}
              </span>
            </p>
          </div>

          {/* controls */}
          <div className="flex items-center gap-2 shrink-0">
            <span
              className={[
                "px-2.5 py-1 text-[10px] font-semibold border",
                sl.warn
                  ? "border-watch/40 bg-watch/8 text-watch"
                  : "border-safe/40 bg-safe/8 text-safe",
              ].join(" ")}
              style={{ borderRadius: "var(--radius-control)" }}
            >
              {sl.text}
            </span>
            <ToolBtn onClick={() => setView((v) => v === "simple" ? "detailed" : "simple")} label={view === "simple" ? "Switch to detailed view" : "Switch to simple view"} active={view === "detailed"}>
              {view === "simple" ? "Detailed View" : "Simple View"}
            </ToolBtn>
            <ToolBtn onClick={() => setDark((v) => !v)} label={dark ? "Switch to light theme" : "Switch to dark theme"}>
              {dark ? "☼ Light mode" : "☾ Dark mode"}
            </ToolBtn>
            <ToolBtn onClick={() => setRefresh((v) => v + 1)} label="Refresh data">
              ↻ Refresh data
            </ToolBtn>
          </div>
        </header>

        {/* ─── Summary bar ─── */}
        <div className="flex flex-wrap items-center gap-x-8 gap-y-2 py-3 border-b border-line">
          <RiskStat count={summary?.HIGH ?? 0}     label="High risk"     colorClass="text-danger" />
          <div className="w-px h-5 bg-line hidden sm:block" />
          <RiskStat count={summary?.MODERATE ?? 0} label="Moderate risk" colorClass="text-watch"  />
          <div className="w-px h-5 bg-line hidden sm:block" />
          <RiskStat count={summary?.LOW ?? 0}      label="Low risk"      colorClass="text-safe"   />
          <div className="w-px h-5 bg-line hidden sm:block" />
          <span className="text-xs text-ink-subtle">
            {summary?.total ?? metadata?.predicted_communes ?? 0} communes predicted
          </span>
          <span className="ml-auto text-xs text-ink-subtle">
            Updated{" "}
            <span className="font-data text-ink">{formatTimestamp(lastUpdated)}</span>
            {status?.model_version && (
              <span className="ml-2 text-ink-subtle">· {status.model_version}</span>
            )}
          </span>
        </div>

        {/* ═══════════════ SIMPLE VIEW ═══════════════ */}
    {view === "simple" && (
      <div className="py-5 space-y-4 view-enter">

        {/* Full-width map */}
        <div className="lg:h-[calc(100vh-180px)] lg:min-h-0">

          <section
            className="border border-line bg-surface overflow-hidden h-full min-h-0 w-full"
            style={{ borderRadius: "var(--radius-control)" }}
          >
            <RiskMap
              geojson={map}
              selectedId={selectedId}
              onSelect={select}
              dark={dark}
              visibleIds={visibleIds}
              priorityOnly={priorityOnly}
            />
          </section>

        </div>

        {/* Filters + action list */}
        <section
          className="border border-line bg-surface overflow-hidden h-full min-h-0"
          style={{ borderRadius: "var(--radius-control)" }}
        >
          <div className="border-b border-line">
            <Filters {...filterProps} />
          </div>

          <ActionList
            items={filteredItems}
            selectedId={selectedId}
            priorityOnly={priorityOnly}
            onSelect={(item) => select(item.commune_id)}
          />
        </section>

      </div>
    )}

        {/* ═══════════════ DETAILED VIEW ═══════════════ */}
        {view === "detailed" && (
          <div className="py-5 space-y-5 view-enter">
            <NationalOverview summary={summary} allItems={allItems} wilayas={wilayas} />

              <div className="grid gap-4 lg:grid-cols-[minmax(0,1.5fr)_340px] lg:h-[calc(100vh-180px)] lg:min-h-0 lg:items-stretch">              
                <section
                className="border border-line bg-surface overflow-hidden h-full min-h-0"
                style={{ borderRadius: "var(--radius-control)" }}
              >
                <div className="border-b border-line">
                  <Filters {...filterProps} />
                </div>
                <RiskMap
                  geojson={map}
                  selectedId={selectedId}
                  onSelect={select}
                  dark={dark}
                  visibleIds={visibleIds}
                  priorityOnly={priorityOnly}
                />
              </section>
              {detailPanel}
            </div>

            <DetailedInsights
              summary={summary}
              status={status}
              wilayas={wilayas}
              history={history}
              communeHistory={communeHistory}
              selected={selectedPrediction}
              allItems={allItems}
            />
          </div>
        )}

        {/* ─── Error ─── */}
        {error && (
          <div
            className="my-4 px-4 py-3 text-xs text-ink border border-danger/30 bg-danger/8"
            style={{ borderRadius: "var(--radius-control)" }}
          >
            {error}
          </div>
        )}
      </div>
    </main>
  );
}

export default function Home() {
  return (
    <Suspense
      fallback={
        <div className="grid min-h-screen place-items-center bg-canvas text-sm text-ink-muted">
          Loading…
        </div>
      }
    >
      <HomeContent />
    </Suspense>
  );
}