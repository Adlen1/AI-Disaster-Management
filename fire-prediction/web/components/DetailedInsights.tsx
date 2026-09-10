"use client";

import { useMemo, useState } from "react";
import type {
  CommunePrediction, DailySummary, HistoricalCommunePoint,
  HistoricalNationalPoint, RunStatus, WilayaStats,
} from "../lib/types";

type Props = {
  summary: DailySummary | null;
  status: RunStatus | null;
  wilayas: WilayaStats[];
  history: HistoricalNationalPoint[];
  communeHistory: HistoricalCommunePoint[];
  selected: CommunePrediction | null;
  allItems: CommunePrediction[];
};

/* ── helpers ── */
function pct(v?: number | null) { return v == null ? "—" : `${Math.round(v * 100)}%`; }
function num(v?: number | null, d = 1) { return v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d); }
function dateLabel(v?: string | null) {
  if (!v) return "—";
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? v : new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short" }).format(d);
}
function avg(items: CommunePrediction[], key: keyof CommunePrediction) {
  const vals = items.map((i) => i[key]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
}
function normalizedUrgency(item: CommunePrediction) {
  return (item.urgency_label ?? "").trim().toUpperCase().replaceAll("_", " ");
}
function isPriority(item: CommunePrediction) {
  const l = normalizedUrgency(item);
  return l === "CRITICAL" || l === "HIGH PRIORITY";
}

const DRIVER_LABELS: Record<string, string> = {
  fire_count: "Recent detected fires", fire_count_3d: "Fires in last 3 days",
  fire_count_7d: "Fires in last 7 days", frp_total: "Fire radiative power",
  FWI: "Fire Weather Index", temp_c: "Temperature", rh: "Relative humidity",
  DMC: "Fuel dryness (shallow)", DC: "Fuel dryness (deep)",
  NDVI: "Vegetation greenness", NBR: "Vegetation burn index",
  burnable_fraction: "Burnable land fraction", forest_fraction: "Forest coverage",
  pop_density_mean: "Population density", road_distance_mean_km: "Distance to nearest road",
  commune_fire_rate: "Historical fire rate", days_since_fire: "Days since last fire",
};
function driverLabel(f: string) { return DRIVER_LABELS[f] ?? f.replaceAll("_", " "); }

function environmentalNote(key: string, value: number | null | undefined): string {
  if (value == null) return "Not available";
  if (key === "temp_c")  return value >= 35 ? "Extreme heat" : value >= 28 ? "Hot — elevated risk" : "Moderate temperature";
  if (key === "rh")      return value <= 25 ? "Very dry air — dangerous" : value <= 40 ? "Dry conditions" : "Adequate humidity";
  if (key === "FWI")     return value >= 30 ? "Very high fire-weather danger" : value >= 15 ? "Elevated danger" : "Moderate";
  if (key === "DMC")     return "Shallow fuel and litter dryness";
  if (key === "DC")      return "Deep fuel moisture — slow to recover";
  if (key === "NDVI")    return value <= 0.2 ? "Sparse / dry vegetation" : "Adequate vegetation cover";
  if (key === "NBR")     return "Detects vegetation stress and past burns";
  if (key === "fire_count") return value > 0 ? "Active fire history detected" : "No recent detections";
  if (key === "burnable_fraction") return "Share of commune that can physically burn";
  if (key === "forest_fraction")   return "Share covered by forest";
  if (key === "pop_density_mean")  return "People potentially exposed to fire";
  if (key === "road_distance_mean_km") return "Affects response and evacuation time";
  return "Model input factor";
}

/* ── UI primitives ── */
const rc = { borderRadius: "var(--radius-control)" };
function Div() { return <div style={{ borderTop: "1px solid var(--line)" }} />; }
function Panel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <article className={`border border-line bg-surface overflow-hidden animate-up ${className}`} style={rc}>{children}</article>;
}
function PanelHead({ children }: { children: React.ReactNode }) {
  return <div className="px-4 py-3 border-b border-line">{children}</div>;
}
function PanelTitle({ children }: { children: React.ReactNode }) {
  return <h3 className="text-sm font-semibold text-ink">{children}</h3>;
}
function Block({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`px-4 py-3 ${className}`}>{children}</div>;
}
function Label({ children }: { children: React.ReactNode }) {
  return <p className="text-[10px] text-ink-subtle uppercase tracking-wide mb-2">{children}</p>;
}

/* ════════════════════════════════════════════════════════
   STATISTICS SECTION (shown when no commune is selected)
   ════════════════════════════════════════════════════════ */

type SortKey = "elevated_share" | "average_urgency" | "HIGH" | "fire_detections";
type RiskFilter = "all" | "HIGH" | "MODERATE" | "LOW" | "priority";

function StatisticsSection({ allItems, wilayas }: { allItems: CommunePrediction[]; wilayas: WilayaStats[] }) {
  const [wilayaSort, setWilayaSort] = useState<SortKey>("elevated_share");
  const [communeRisk, setCommuneRisk] = useState<RiskFilter>("all");
  const [communeWilaya, setCommuneWilaya] = useState("all");
  const [communeSort, setCommuneSort] = useState<"urgency" | "prob_high">("urgency");

  const wilayasAvailable = useMemo(
    () => [...new Set(allItems.map((i) => i.wilaya_name).filter(Boolean))].sort(),
    [allItems],
  );

  const sortedWilayas = useMemo(() => {
    return [...wilayas].sort((a, b) => {
      if (wilayaSort === "elevated_share")  return (b.elevated_share ?? 0) - (a.elevated_share ?? 0);
      if (wilayaSort === "average_urgency") return (b.average_urgency ?? 0) - (a.average_urgency ?? 0);
      if (wilayaSort === "HIGH")            return (b.HIGH ?? 0) - (a.HIGH ?? 0);
      if (wilayaSort === "fire_detections") return (b.fire_detections ?? 0) - (a.fire_detections ?? 0);
      return 0;
    });
  }, [wilayas, wilayaSort]);

  const filteredCommunes = useMemo(() => {
    return allItems
      .filter((item) => {
        if (communeRisk === "priority") return isPriority(item);
        if (communeRisk !== "all") return item.risk_label.toUpperCase() === communeRisk;
        return true;
      })
      .filter((item) => communeWilaya === "all" || item.wilaya_name === communeWilaya)
      .sort((a, b) =>
        communeSort === "urgency"
          ? b.urgency_score - a.urgency_score
          : (b.prob_high ?? 0) - (a.prob_high ?? 0),
      );
  }, [allItems, communeRisk, communeWilaya, communeSort]);

  const selectCls = "px-2.5 py-1.5 text-xs border border-line bg-surface text-ink focus:outline-none focus:ring-2 focus:ring-brand/20";
  const sortBtnCls = (active: boolean) =>
    `px-2.5 py-1 text-xs font-semibold border transition-colors ${active ? "border-brand bg-brand text-white" : "border-line bg-surface text-ink hover:border-brand/40"}`;

  return (
    <div className="space-y-4">

      {/* ── Wilaya rankings ── */}
      <Panel>
        <PanelHead>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <PanelTitle>Wilaya overview</PanelTitle>
            <div className="flex flex-wrap gap-2 items-center">
              <span className="text-[10px] text-ink-subtle">Sort by:</span>
              {([
                ["elevated_share",  "Elevated share"],
                ["HIGH",            "High-risk count"],
                ["average_urgency", "Avg urgency"],
                ["fire_detections", "Fire detections"],
              ] as [SortKey, string][]).map(([k, l]) => (
                <button key={k} type="button" style={rc} className={sortBtnCls(wilayaSort === k)} onClick={() => setWilayaSort(k)}>{l}</button>
              ))}
            </div>
          </div>
        </PanelHead>
        <div className="overflow-y-auto" style={{ maxHeight: 340 }}>
          {sortedWilayas.length === 0 && (
            <Block><p className="text-sm text-ink-muted">No wilaya data available.</p></Block>
          )}
          {sortedWilayas.map((w, idx) => {
            const share = w.elevated_share ?? (w.total ? (w.HIGH + w.MODERATE) / w.total : 0);
            return (
              <div key={w.wilaya_name} className="flex items-center gap-3 px-4 py-2.5 border-b border-line last:border-b-0 row-hover hover:bg-canvas">
                <span className="font-data text-xs text-ink-subtle w-5 shrink-0">{idx + 1}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-semibold text-ink truncate">{w.wilaya_name}</p>
                  <div className="mt-1 h-1.5 bg-surface-muted" style={{ borderRadius: 2 }}>
                    <div className="h-1.5 bg-brand bar-animate" style={{ width: `${Math.min(100, share * 100)}%`, borderRadius: 2 }} />
                  </div>
                </div>
                <div className="shrink-0 text-right">
                  <p className="font-data text-xs font-medium text-ink">{pct(w.elevated_share)}</p>
                  <p className="text-[10px] text-ink-subtle">elevated</p>
                </div>
                <div className="shrink-0 text-right hidden sm:block">
                  <p className="font-data text-xs text-danger">{w.HIGH}</p>
                  <p className="text-[10px] text-ink-subtle">HIGH</p>
                </div>
                <div className="shrink-0 text-right hidden md:block">
                  <p className="font-data text-xs text-ink">{num(w.average_urgency, 0)}</p>
                  <p className="text-[10px] text-ink-subtle">avg urgency</p>
                </div>
              </div>
            );
          })}
        </div>
      </Panel>

      {/* ── Commune list ── */}
      <Panel>
        <PanelHead>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <PanelTitle>Commune list</PanelTitle>
            <div className="flex flex-wrap gap-2 items-center">
              <select value={communeRisk} onChange={(e) => setCommuneRisk(e.target.value as RiskFilter)} className={selectCls} style={rc}>
                <option value="all">All risk levels</option>
                <option value="HIGH">High only</option>
                <option value="MODERATE">Moderate only</option>
                <option value="LOW">Low only</option>
                <option value="priority">Priority only</option>
              </select>
              <select value={communeWilaya} onChange={(e) => setCommuneWilaya(e.target.value)} className={selectCls} style={rc}>
                <option value="all">All wilayas</option>
                {wilayasAvailable.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
              <button type="button" style={rc} className={sortBtnCls(communeSort === "urgency")} onClick={() => setCommuneSort("urgency")}>By urgency</button>
              <button type="button" style={rc} className={sortBtnCls(communeSort === "prob_high")} onClick={() => setCommuneSort("prob_high")}>By P(HIGH)</button>
            </div>
          </div>
        </PanelHead>
        <div className="overflow-y-auto" style={{ maxHeight: 400 }}>
          {filteredCommunes.length === 0 && (
            <Block><p className="text-sm text-ink-muted">No communes match the selected filters.</p></Block>
          )}
          {filteredCommunes.map((item) => {
            const riskCls = item.risk_label.toUpperCase() === "HIGH" ? "text-danger" : item.risk_label.toUpperCase() === "MODERATE" ? "text-watch" : "text-safe";
            const barCls  = item.risk_label.toUpperCase() === "HIGH" ? "risk-bar-high" : item.risk_label.toUpperCase() === "MODERATE" ? "risk-bar-mod" : "risk-bar-low";
            return (
              <div key={item.commune_id} className={`flex items-center gap-3 px-4 py-2.5 border-b border-line last:border-b-0 ${barCls}`}>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-semibold text-ink truncate">
                    {item.commune_name} <span className="font-normal text-ink-muted">· {item.wilaya_name}</span>
                  </p>
                  <p className="text-[10px] text-ink-subtle mt-0.5">
                    {isPriority(item) && <span className="text-brand font-semibold mr-1.5">Priority</span>}
                    {normalizedUrgency(item) || ""}
                  </p>
                </div>
                <div className="shrink-0 text-right">
                  <p className={`font-data text-xs font-semibold ${riskCls}`}>{item.risk_label.toUpperCase()}</p>
                  <p className="text-[10px] text-ink-subtle">risk</p>
                </div>
                <div className="shrink-0 text-right">
                  <p className="font-data text-xs text-ink">{pct(item.prob_high)}</p>
                  <p className="text-[10px] text-ink-subtle">P(HIGH)</p>
                </div>
                <div className="shrink-0 text-right">
                  <p className="font-data text-xs text-ink">{Math.round(item.urgency_score)}</p>
                  <p className="text-[10px] text-ink-subtle">urgency</p>
                </div>
              </div>
            );
          })}
        </div>
        <div className="px-4 py-2 border-t border-line bg-canvas">
          <p className="text-[10px] text-ink-subtle">{filteredCommunes.length} commune{filteredCommunes.length !== 1 ? "s" : ""} shown</p>
        </div>
      </Panel>

    </div>
  );
}

/* ════════════════════════════════════════════════════════
   COMMUNE-SELECTED SECTIONS
   ════════════════════════════════════════════════════════ */

function ProbabilityDistribution({ selected }: { selected: CommunePrediction }) {
  const bars = [
    { label: "Low",      v: selected.prob_low,      cls: "bg-safe"   },
    { label: "Moderate", v: selected.prob_moderate,  cls: "bg-watch"  },
    { label: "High",     v: selected.prob_high,      cls: "bg-danger" },
  ];
  return (
    <Panel>
      <PanelHead><PanelTitle>Risk probability breakdown</PanelTitle></PanelHead>
      <Block>
        <div className="flex h-6 overflow-hidden" style={rc}>
          {bars.map((b) => (
            <div key={b.label} className={`${b.cls} flex items-center justify-center`} style={{ width: `${Math.max(0, (b.v ?? 0) * 100)}%` }}>
              {(b.v ?? 0) >= 0.1 && <span className="text-[9px] font-semibold text-white">{b.label}</span>}
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-2">
          {bars.map((b) => (
            <div key={b.label} className="text-center">
              <p className="font-data text-lg font-medium text-ink">{pct(b.v)}</p>
              <p className="text-[10px] text-ink-subtle">{b.label}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-ink-muted mt-3 leading-5">
          The model outputs a probability for each class. The one with the highest probability is the predicted class.
          HIGH is triggered when its probability is at least 39% — a deliberately sensitive threshold to avoid missing real fires.
        </p>
        <div className="mt-3 grid grid-cols-3 gap-2 border-t border-line pt-3 text-center">
          <div>
            <p className="text-[10px] text-ink-subtle">Predicted class</p>
            <p className="font-data text-sm font-medium text-ink">{selected.risk_label}</p>
          </div>
          <div>
            <p className="text-[10px] text-ink-subtle">HIGH threshold</p>
            <p className="font-data text-sm font-medium text-ink">≥ 0.39</p>
          </div>
          <div>
            <p className="text-[10px] text-ink-subtle">Current P(HIGH)</p>
            <p className="font-data text-sm font-medium text-ink">{num(selected.prob_high, 3)}</p>
          </div>
        </div>
      </Block>
    </Panel>
  );
}

function Drivers({ selected }: { selected: CommunePrediction }) {
  const drivers = [...(selected.top_drivers ?? [])].slice(0, 7);
  if (!drivers.length) return null;
  const maxVal = Math.max(...drivers.map((d) => Math.abs(d.shap)), 0.001);
  const strongest = drivers.find((d) => d.shap >= 0) ?? drivers[0];
  return (
    <Panel>
      <PanelHead>
        <PanelTitle>Why this prediction? — SHAP factor contributions</PanelTitle>
      </PanelHead>
      <Block>
        <p className="text-xs text-ink-muted leading-5 mb-4">
          SHAP values measure how much each factor pushed the model toward (red) or away from (green) HIGH risk.
          Longer bar = stronger influence. The values come directly from the model — nothing is invented.
        </p>
        <div className="space-y-3">
          {drivers.map((d) => {
            const pos = d.shap >= 0;
            const w = Math.max(8, (Math.abs(d.shap) / maxVal) * 100);
            return (
              <div key={d.feature}>
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="text-xs text-ink">{driverLabel(d.feature)}</span>
                  <span className={`font-data text-[11px] shrink-0 ${pos ? "text-danger" : "text-safe"}`}>
                    {d.shap >= 0 ? "+" : ""}{d.shap.toFixed(3)}
                  </span>
                </div>
                <div className="h-1.5 bg-surface-muted" style={{ borderRadius: 2 }}>
                  <div className={`h-1.5 bar-animate ${pos ? "bg-danger/70" : "bg-safe/70"}`} style={{ width: `${w}%`, borderRadius: 2 }} />
                </div>
                <p className="text-[10px] text-ink-subtle mt-1 leading-4">
                  Observed value: <span className="font-data">{num(d.value, 2)}</span>. This {pos ? "increases" : "decreases"} predicted risk.
                </p>
              </div>
            );
          })}
        </div>
        {strongest && (
          <div className="mt-4 p-3 bg-canvas border border-line text-xs leading-5 text-ink" style={rc}>
            <strong>{driverLabel(strongest.feature)}</strong> is the strongest driver toward HIGH risk for this commune
            (SHAP: {strongest.shap >= 0 ? "+" : ""}{strongest.shap.toFixed(3)}, observed value: {num(strongest.value, 2)}).
          </div>
        )}
      </Block>
    </Panel>
  );
}

function EnvironmentalEvidence({ selected }: { selected: CommunePrediction }) {
  const cards: { key: keyof CommunePrediction; label: string; unit?: string; digits?: number }[] = [
    { key: "temp_c",              label: "Temperature",          unit: "°C", digits: 1 },
    { key: "rh",                  label: "Humidity",             unit: "%",  digits: 0 },
    { key: "FWI",                 label: "Fire Weather Index",   digits: 1  },
    { key: "DMC",                 label: "Fuel dryness",         digits: 0  },
    { key: "DC",                  label: "Deep fuel dryness",    digits: 0  },
    { key: "NDVI",                label: "Vegetation greenness", digits: 2  },
    { key: "NBR",                 label: "Vegetation burn idx",  digits: 2  },
    { key: "fire_count",          label: "Recent fires",         digits: 0  },
    { key: "burnable_fraction",   label: "Burnable land",        digits: 2  },
    { key: "forest_fraction",     label: "Forest fraction",      digits: 2  },
    { key: "pop_density_mean",    label: "Population density",   unit: "/km²", digits: 0 },
    { key: "road_distance_mean_km", label: "Road distance",      unit: " km",  digits: 1 },
  ];
  return (
    <Panel>
      <PanelHead><PanelTitle>Current conditions — all available inputs</PanelTitle></PanelHead>
      <Block>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
          {cards.map((c) => {
            const raw = selected[c.key] as number | null | undefined;
            const value = raw == null ? "—" : c.key === "burnable_fraction" || c.key === "forest_fraction"
              ? pct(raw)
              : `${num(raw, c.digits ?? 1)}${c.unit ?? ""}`;
            return (
              <div key={String(c.key)} className="p-3 border border-line bg-canvas" style={rc}>
                <p className="font-data text-sm font-medium text-ink">{value}</p>
                <p className="text-[11px] text-ink-muted mt-0.5">{c.label}</p>
                <p className="text-[10px] text-ink-subtle mt-1 leading-3.5">{environmentalNote(String(c.key), raw)}</p>
              </div>
            );
          })}
        </div>
      </Block>
    </Panel>
  );
}

function TemporalHistory({ history }: { history: HistoricalCommunePoint[] }) {
  const pts = history.filter((p) => p.prob_high != null).slice(-30);
  const maxV = Math.max(...pts.map((p) => p.prob_high ?? 0), 0.01);
  const W = 300; const H = 80;
  const step = W / Math.max(pts.length - 1, 1);
  const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${(i * step).toFixed(1)} ${(H - ((p.prob_high ?? 0) / maxV) * H).toFixed(1)}`).join(" ");
  const areaD = `${pathD} L ${W} ${H} L 0 ${H} Z`;
  const threshY = H - (0.39 / maxV) * H;

  return (
    <Panel>
      <PanelHead><PanelTitle>Recent P(HIGH) evolution</PanelTitle></PanelHead>
      <Block>
        {pts.length === 0 ? (
          <p className="text-sm text-ink-muted">No dated commune archive points were returned.</p>
        ) : (
          <>
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" height={H} aria-hidden="true" preserveAspectRatio="none">
              {threshY > 0 && threshY < H && (
                <line x1="0" y1={threshY} x2={W} y2={threshY} stroke="var(--danger)" strokeWidth="1" strokeDasharray="4 3" opacity="0.7" />
              )}
              <path d={areaD} fill="var(--brand)" opacity="0.1" />
              <path d={pathD} fill="none" stroke="var(--brand)" strokeWidth="2" strokeLinejoin="round" />
            </svg>

            {/* legend */}
            <div className="flex flex-wrap items-center gap-x-5 gap-y-1 mt-2 text-[10px] text-ink-subtle">
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-5 border-t-2" style={{ borderColor: "var(--brand)" }} />
                P(HIGH) — probability of HIGH risk
              </span>
              <span className="flex items-center gap-1.5">
                <svg width="16" height="6" viewBox="0 0 16 6">
                  <line x1="0" y1="3" x2="16" y2="3" stroke="var(--danger)" strokeWidth="1.2" strokeDasharray="4 3" />
                </svg>
                HIGH threshold (0.39)
              </span>
            </div>
            <div className="flex justify-between mt-1 text-[10px] text-ink-subtle">
              <span>{dateLabel(pts[0]?.date)}</span>
              <span>{dateLabel(pts.at(-1)?.date)}</span>
            </div>
            <p className="text-[10px] text-ink-subtle mt-2 leading-4">
              When the green line crosses the dashed red line, the model predicted HIGH risk for that day.
              Based on {pts.length} archived prediction records.
            </p>
          </>
        )}
      </Block>
    </Panel>
  );
}

function OperationalContext({ selected }: { selected: CommunePrediction }) {
  const cards = [
    { label: "Population density",  value: selected.pop_density_mean == null ? "—" : `${num(selected.pop_density_mean, 0)}/km²`,       note: "People potentially in the path of a fire" },
    { label: "Burnable land",       value: pct(selected.burnable_fraction),                                                             note: "Share of commune land that can physically burn" },
    { label: "Distance to road",    value: selected.road_distance_mean_km == null ? "—" : `${num(selected.road_distance_mean_km, 1)} km`, note: "Average distance to nearest road — affects response time" },
    { label: "Detected fires",      value: selected.fire_count == null ? "—" : num(selected.fire_count, 0),                             note: "Recent fire detections in this commune" },
  ];
  return (
    <Panel>
      <PanelHead><PanelTitle>Why does urgency matter here?</PanelTitle></PanelHead>
      <Block>
        <p className="text-xs text-ink-muted leading-5 mb-3">
          Urgency combines predicted risk with operational reality: how many people live here, how reachable it is, and how much can burn.
          Two communes can both be HIGH risk but differ greatly in urgency — a remote uninhabited area vs. a densely populated one.
        </p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {cards.map((c) => (
            <div key={c.label} className="p-3 border border-line bg-canvas" style={rc}>
              <p className="font-data text-sm font-medium text-ink">{c.value}</p>
              <p className="text-[11px] text-ink-muted mt-0.5">{c.label}</p>
              <p className="text-[10px] text-ink-subtle mt-1 leading-3.5">{c.note}</p>
            </div>
          ))}
        </div>
      </Block>
    </Panel>
  );
}

function PeerComparison({ selected, items }: { selected: CommunePrediction; items: CommunePrediction[] }) {
  const wilayaItems = items.filter((i) => i.wilaya_name === selected.wilaya_name);
  const rows = [
    { label: "P(HIGH)",            s: selected.prob_high,         w: avg(wilayaItems, "prob_high"),        n: avg(items, "prob_high"),        fmt: (v: number | null) => pct(v) },
    { label: "Fire Weather (FWI)", s: selected.FWI,               w: avg(wilayaItems, "FWI"),              n: avg(items, "FWI"),              fmt: (v: number | null) => num(v, 1) },
    { label: "Recent fires",       s: selected.fire_count,        w: avg(wilayaItems, "fire_count"),       n: avg(items, "fire_count"),       fmt: (v: number | null) => num(v, 1) },
    { label: "Population/km²",     s: selected.pop_density_mean,  w: avg(wilayaItems, "pop_density_mean"), n: avg(items, "pop_density_mean"), fmt: (v: number | null) => num(v, 0) },
    { label: "Burnable land",      s: selected.burnable_fraction, w: avg(wilayaItems, "burnable_fraction"),n: avg(items, "burnable_fraction"), fmt: (v: number | null) => pct(v) },
  ];
  return (
    <Panel>
      <PanelHead><PanelTitle>Commune vs wilaya vs national average</PanelTitle></PanelHead>
      <Block>
        <p className="text-xs text-ink-muted mb-3 leading-5">
          Compares this commune's key figures against the average across {selected.wilaya_name} wilaya and across all Algeria.
          Helps you judge whether this commune is unusually elevated relative to its surroundings.
          Wilaya and national values are averages computed from the current prediction run.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-left text-xs">
            <thead>
              <tr className="border-b border-line">
                <th className="py-2 text-ink-subtle font-normal">Measure</th>
                <th className="py-2 font-semibold text-ink">{selected.commune_name}</th>
                <th className="py-2 text-ink">{selected.wilaya_name}</th>
                <th className="py-2 text-ink">Algeria</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.label} className="border-b border-line/60">
                  <td className="py-2 text-ink-muted">{r.label}</td>
                  <td className="py-2 font-data font-semibold text-ink">{r.fmt(r.s)}</td>
                  <td className="py-2 font-data text-ink">{r.fmt(r.w)}</td>
                  <td className="py-2 font-data text-ink">{r.fmt(r.n)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Block>
    </Panel>
  );
}

function ReliabilityAndFlags({ selected }: { selected: CommunePrediction }) {
  const flags: { label: string; meaning: string; action: string }[] = [];
  if (selected.borderline) flags.push({
    label: "Borderline result",
    meaning:
      "The model's probability score is very close to the 0.39 threshold that separates MODERATE from HIGH. In practice this means the prediction is uncertain — a change of a few percentage points in P(HIGH) would flip the class. This can happen when conditions are genuinely ambiguous or when input data has slight variations.",
    action: "Treat this commune as HIGH risk until field conditions confirm otherwise. Do not rely solely on the predicted class.",
  });
  if (selected.low_confidence) flags.push({
    label: "Low confidence — historical miss rate",
    meaning:
      "The model has previously missed HIGH-risk events in this specific area. This flag is set based on validation results from past prediction runs — it does not mean the current prediction is wrong, but it means the model is less reliable here than elsewhere.",
    action: "Apply heightened caution regardless of the current risk label. Consider this area as higher priority.",
  });
  const quality = selected.data_quality_status?.toLowerCase() ?? "";
  if (quality.includes("stale") || quality.includes("warning") || quality.includes("caution")) flags.push({
    label: "Data quality warning",
    meaning: `The backend reported a data quality issue: "${selected.data_quality_status}". Wildfire predictions depend on recent weather observations and satellite imagery. When that data is delayed, missing, or of poor quality, the model is working with outdated or incomplete information — and the prediction is less reliable as a result.`,
    action: "Check how many days the weather data is delayed (shown at the top of the page). If staleness is high, treat all predictions with additional caution.",
  });

  return (
    <Panel>
      <PanelHead>
        <div className="flex items-center justify-between gap-3">
          <PanelTitle>Prediction reliability & flags</PanelTitle>
          {flags.length === 0
            ? <span className="text-[10px] font-semibold text-safe bg-safe/10 border border-safe/30 px-2 py-0.5" style={rc}>Reliable</span>
            : <span className="text-[10px] font-semibold text-watch bg-watch/10 border border-watch/30 px-2 py-0.5" style={rc}>{flags.length} flag{flags.length > 1 ? "s" : ""}</span>
          }
        </div>
      </PanelHead>
      <Block>
        {flags.length === 0 ? (
          <p className="text-xs text-safe flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-safe inline-block" />
            No reliability flags for this prediction. No borderline result, no known model weakness in this area, no data quality issues.
          </p>
        ) : (
          <div className="space-y-4">
            {flags.map((f) => (
              <div key={f.label} className="border border-watch/30 bg-watch/5 p-3" style={rc}>
                <p className="text-xs font-semibold text-watch">{f.label}</p>
                <p className="text-xs text-ink-muted mt-1.5 leading-5">{f.meaning}</p>
                <p className="text-xs font-semibold text-ink mt-2 leading-5">→ {f.action}</p>
              </div>
            ))}
            <p className="text-[10px] text-ink-subtle leading-4">
              Flags do not invalidate a prediction — they are indicators that this result should be treated with additional care.
              A "Reliable" prediction can still be wrong; a flagged prediction can still be correct.
            </p>
          </div>
        )}
      </Block>
    </Panel>
  );
}

/* ══════════════════════════════════ */
export default function DetailedInsights({
  summary, status, wilayas, history, communeHistory, selected, allItems,
}: Props) {
  if (!selected) {
    return (
      <section className="space-y-4">
        <StatisticsSection allItems={allItems} wilayas={wilayas} />
      </section>
    );
  }

  /* When a commune is selected, the scrollable detail panel already shows:
     probability breakdown, SHAP drivers, current conditions, urgency context,
     P(HIGH) sparkline, and reliability flags.
     Only show the peer comparison here — it's the one thing the panel doesn't have. */
  return (
    <section className="space-y-4">
      <PeerComparison selected={selected} items={allItems} />
    </section>
  );
}

/* ── NationalOverview removed — summary bar in page.tsx handles this ── */
export function NationalOverview(_: {
  summary: DailySummary | null;
  allItems: CommunePrediction[];
  wilayas: WilayaStats[];
}) {
  return null;
}