"use client";

import type { CommunePrediction, CommuneResponse, HistoricalCommunePoint } from "../lib/types";
import { reasonCategory } from "./ActionList";

type Props = {
  prediction: CommuneResponse | null;
  history: HistoricalCommunePoint[];
  loading: boolean;
  onClose: () => void;
};

/* ── helpers ── */
function pct(v?: number | null) {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}
function num(v?: number | null, digits = 1) {
  return v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);
}
function formatDate(v?: string | null) {
  if (!v) return "Unavailable";
  const d = new Date(v);
  return Number.isNaN(d.getTime())
    ? v
    : new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric" }).format(d);
}
function normalizedUrgency(item: CommunePrediction) {
  return (item.urgency_label ?? "").trim().toUpperCase().replaceAll("_", " ");
}

const RISK_STYLES: Record<string, { bg: string; text: string }> = {
  HIGH:     { bg: "bg-danger", text: "text-white" },
  MODERATE: { bg: "bg-watch",  text: "text-white" },
  LOW:      { bg: "bg-safe",   text: "text-white" },
};

/* ── Flags ── */
type FlagInfo = { label: string; meaning: string; action: string };
function getFlags(item: CommunePrediction): FlagInfo[] {
  const flags: FlagInfo[] = [];
  if (item.borderline) flags.push({
    label: "Borderline result",
    meaning:
      "The model's confidence is sitting very close to the line between HIGH and MODERATE risk. A small change in conditions — a few degrees warmer, a bit drier — could push this into a different category.",
    action: "Treat this commune as HIGH risk until field conditions are confirmed.",
  });
  if (item.low_confidence) flags.push({
    label: "Low confidence area",
    meaning:
      "The model has historically been less reliable for this specific area — it has missed HIGH-risk events here before. This doesn't change the prediction, but it means you should trust it less.",
    action: "Apply extra caution even if the current risk label appears moderate.",
  });
  const quality = item.data_quality_status?.toLowerCase() ?? "";
  if (quality.includes("stale") || quality.includes("warning") || quality.includes("caution"))
    flags.push({
      label: "Data quality warning",
      meaning: `The input data for this prediction has a quality issue: "${item.data_quality_status}". Predictions depend on recent weather and satellite data — when that data is delayed or missing, the result is less reliable.`,
      action: "Check the weather staleness indicator at the top of the page.",
    });
  return flags;
}

function reliabilityChip(item: CommunePrediction) {
  const flags = getFlags(item);
  if (!flags.length) return { label: "Reliable", cls: "text-safe bg-safe/10 border-safe/30" };
  return { label: `${flags.length} flag${flags.length > 1 ? "s" : ""}`, cls: "text-watch bg-watch/10 border-watch/30" };
}

function recommendation(item: CommunePrediction) {
  const urg = normalizedUrgency(item);
  if (urg === "CRITICAL") return "Immediate operational attention required.";
  if (item.risk_label.toUpperCase() === "HIGH") return "Prioritize for assessment and active monitoring.";
  if (item.risk_label.toUpperCase() === "MODERATE") return "Maintain enhanced watch; reassess if conditions deteriorate.";
  return "No elevated risk detected. Continue routine monitoring.";
}

/* ── SHAP drivers ── */
const DRIVER_LABELS: Record<string, string> = {
  fire_count: "Recent detected fires",
  fire_count_3d: "Fires in last 3 days",
  fire_count_7d: "Fires in last 7 days",
  frp_total: "Fire radiative power",
  FWI: "Fire Weather Index",
  temp_c: "Temperature",
  rh: "Relative humidity",
  DMC: "Fuel dryness (shallow)",
  DC: "Fuel dryness (deep)",
  NDVI: "Vegetation greenness",
  NBR: "Vegetation burn index",
  burnable_fraction: "Burnable land fraction",
  forest_fraction: "Forest coverage",
  pop_density_mean: "Population density",
  road_distance_mean_km: "Distance to nearest road",
  commune_fire_rate: "Historical fire rate",
  days_since_fire: "Days since last fire",
};
function driverLabel(f: string) {
  return DRIVER_LABELS[f] ?? f.replaceAll("_", " ");
}
function driverExplain(f: string, value: number, shap: number): string {
  const dir = shap >= 0 ? "increases" : "decreases";
  const map: Record<string, (v: number) => string> = {
    FWI:          (v) => `FWI of ${num(v, 0)} — ${v >= 30 ? "very high fire-weather danger" : v >= 15 ? "elevated" : "moderate conditions"}. This ${dir} predicted risk.`,
    temp_c:       (v) => `Temperature of ${num(v, 1)}°C — ${v >= 35 ? "extreme heat" : v >= 28 ? "hot" : "moderate"}. This ${dir} predicted risk.`,
    rh:           (v) => `Humidity of ${num(v, 0)}% — ${v <= 25 ? "very dry air" : v <= 40 ? "dry conditions" : "adequate humidity"}. This ${dir} predicted risk.`,
    fire_count:   (v) => `${num(v, 0)} fire${v === 1 ? "" : "s"} detected recently. This ${dir} predicted risk.`,
    fire_count_3d:(v) => `${num(v, 0)} fire${v === 1 ? "" : "s"} detected in the last 3 days. This ${dir} predicted risk.`,
    fire_count_7d:(v) => `${num(v, 0)} fire${v === 1 ? "" : "s"} detected in the last 7 days. This ${dir} predicted risk.`,
    NDVI:         (v) => `Vegetation greenness (NDVI) of ${num(v, 2)} — ${v <= 0.2 ? "sparse or dry vegetation" : "adequate cover"}. This ${dir} predicted risk.`,
    NBR:          (v) => `Burn index (NBR) of ${num(v, 2)} — detects vegetation stress and past burns. This ${dir} predicted risk.`,
    burnable_fraction: (v) => `${pct(v)} of this commune's land can burn. This ${dir} predicted risk.`,
    pop_density_mean:  (v) => `${num(v, 0)} people/km². More people exposed ${dir} urgency.`,
    road_distance_mean_km: (v) => `${num(v, 1)} km to nearest road — affects response time. This ${dir} predicted risk.`,
  };
  return map[f]?.(value) ?? `${driverLabel(f)}: observed value ${num(value, 2)}. This ${dir} predicted risk.`;
}

/* ── UI primitives ── */
const rc = { borderRadius: "var(--radius-control)" };
function Div() { return <div style={{ borderTop: "1px solid var(--line)" }} />; }
function Block({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`px-4 py-3 ${className}`}>{children}</div>;
}
function Label({ children }: { children: React.ReactNode }) {
  return <p className="text-[10px] text-ink-subtle uppercase tracking-wide mb-2">{children}</p>;
}
function MiniCard({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="px-3 py-2 border border-line bg-canvas" style={rc}>
      <p className="font-data text-sm font-medium text-ink">{value}</p>
      <p className="text-[11px] text-ink-muted mt-0.5">{label}</p>
      <p className="text-[10px] text-ink-subtle mt-1 leading-3.5">{note}</p>
    </div>
  );
}

/* ── Probability bar ── */
function ProbBar({ item }: { item: CommunePrediction }) {
  const bars = [
    { label: "Low", v: item.prob_low, cls: "bg-safe" },
    { label: "Mod", v: item.prob_moderate, cls: "bg-watch" },
    { label: "High", v: item.prob_high, cls: "bg-danger" },
  ];
  return (
    <div>
      <div className="flex h-5 overflow-hidden" style={rc}>
        {bars.map((b) => (
          <div key={b.label} className={`${b.cls} bar-animate flex items-center justify-center`} style={{ width: `${Math.max(0, (b.v ?? 0) * 100)}%` }}>
            {(b.v ?? 0) >= 0.12 && <span className="text-[9px] font-semibold text-white">{b.label}</span>}
          </div>
        ))}
      </div>
      <div className="flex justify-between mt-1">
        {bars.map((b) => <span key={b.label} className="font-data text-xs text-ink">{pct(b.v)}</span>)}
      </div>
      <p className="text-[10px] text-ink-subtle mt-1.5 leading-4">
        The model assigns a probability to each risk class. The highest one wins. HIGH is triggered when P(HIGH) ≥ 39%.
      </p>
    </div>
  );
}

/* ── SHAP chart ── */
function ShapDrivers({ item }: { item: CommunePrediction }) {
  const drivers = [...(item.top_drivers ?? [])].slice(0, 6);
  if (!drivers.length) return <p className="text-xs text-ink-muted">No driver information returned.</p>;
  const maxVal = Math.max(...drivers.map((d) => Math.abs(d.shap)), 0.001);
  return (
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
            <p className="text-[10px] text-ink-subtle mt-1 leading-4">{driverExplain(d.feature, d.value, d.shap)}</p>
          </div>
        );
      })}
      <p className="text-[10px] text-ink-subtle pt-2 border-t border-line leading-4">
        Red bars push toward HIGH risk. Green bars push away from it. Longer bar = stronger influence on this prediction.
      </p>
    </div>
  );
}

/* ── Sparkline ── */
function Sparkline({ history }: { history: HistoricalCommunePoint[] }) {
  const pts = history.filter((p) => p.prob_high != null).slice(-30);
  if (!pts.length) return <p className="text-xs text-ink-muted">No history available.</p>;
  const vals = pts.map((p) => p.prob_high ?? 0);
  const maxV = Math.max(...vals, 0.01);
  const W = 100; const H = 44;
  const step = W / Math.max(vals.length - 1, 1);
  const pathD = vals.map((v, i) => `${i === 0 ? "M" : "L"} ${(i * step).toFixed(1)} ${(H - (v / maxV) * H).toFixed(1)}`).join(" ");
  const areaD = `${pathD} L ${W} ${H} L 0 ${H} Z`;
  const threshY = H - (0.39 / maxV) * H;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" height={H} aria-hidden="true" preserveAspectRatio="none">
        {threshY > 0 && threshY < H && (
          <line x1="0" y1={threshY} x2={W} y2={threshY} stroke="var(--danger)" strokeWidth="0.6" strokeDasharray="3 2" opacity="0.6" />
        )}
        <path d={areaD} fill="var(--brand)" opacity="0.1" />
        <path d={pathD} fill="none" stroke="var(--brand)" strokeWidth="1.5" strokeLinejoin="round" />
      </svg>
      <div className="flex items-center justify-between mt-1 text-[9px] text-ink-subtle">
        <span>{pts[0]?.date?.slice(5)}</span>
        <span className="flex items-center gap-1">
          <svg width="12" height="6" viewBox="0 0 12 6"><line x1="0" y1="3" x2="12" y2="3" stroke="var(--danger)" strokeWidth="1" strokeDasharray="3 2" /></svg>
          HIGH threshold (0.39)
        </span>
        <span>{pts.at(-1)?.date?.slice(5)}</span>
      </div>
      <p className="text-[10px] text-ink-subtle mt-1.5 leading-4">
        When the green line crosses the dashed red line, the model predicted HIGH risk for that day.
      </p>
    </div>
  );
}

/* ── Conditions grid ── */
function CurrentConditions({ item }: { item: CommunePrediction }) {
  const cards = [
    { k: "t",   label: "Temperature",  value: item.temp_c == null ? "—" : `${num(item.temp_c, 1)}°C`,    note: item.temp_c != null && item.temp_c >= 35 ? "Extreme heat" : item.temp_c != null && item.temp_c >= 28 ? "Hot" : "Moderate" },
    { k: "rh",  label: "Humidity",     value: item.rh == null ? "—" : `${num(item.rh, 0)}%`,             note: item.rh != null && item.rh <= 25 ? "Very dry air — danger" : item.rh != null && item.rh <= 40 ? "Dry" : "Adequate" },
    { k: "fwi", label: "Fire weather", value: item.FWI == null ? "—" : num(item.FWI, 1),                 note: item.FWI != null && item.FWI >= 30 ? "Very high danger" : item.FWI != null && item.FWI >= 15 ? "Elevated" : "Moderate" },
    { k: "fc",  label: "Recent fires", value: item.fire_count == null ? "—" : String(item.fire_count),   note: item.fire_count ? "Active fire history" : "No recent detections" },
    { k: "dmc", label: "Fuel dryness", value: item.DMC == null ? "—" : num(item.DMC, 0),                 note: "How dry the vegetation and forest floor are" },
    { k: "veg", label: "Vegetation",   value: item.NDVI == null ? "—" : num(item.NDVI, 2),               note: item.NDVI != null && item.NDVI <= 0.2 ? "Sparse/dry" : "Adequate cover" },
    { k: "b",   label: "Burnable",     value: pct(item.burnable_fraction),                               note: "Share of this commune that can burn" },
    { k: "rd",  label: "Road access",  value: item.road_distance_mean_km == null ? "—" : `${num(item.road_distance_mean_km, 1)} km`, note: "Avg distance to nearest road" },
  ];
  return (
    <div className="grid grid-cols-2 gap-2">
      {cards.map((c) => <MiniCard key={c.k} label={c.label} value={c.value} note={c.note} />)}
    </div>
  );
}

/* ── Flags section ── */
function Flags({ item }: { item: CommunePrediction }) {
  const flags = getFlags(item);
  if (!flags.length) return (
    <p className="text-xs text-safe flex items-center gap-1.5">
      <span className="h-1.5 w-1.5 rounded-full bg-safe inline-block" />
      No reliability flags — this prediction is in good standing.
    </p>
  );
  return (
    <div className="space-y-3">
      {flags.map((f) => (
        <div key={f.label} className="border border-watch/30 bg-watch/5 p-3" style={rc}>
          <p className="text-xs font-semibold text-watch">{f.label}</p>
          <p className="text-xs text-ink-muted mt-1.5 leading-5">{f.meaning}</p>
          <p className="text-xs font-semibold text-ink mt-2">→ {f.action}</p>
        </div>
      ))}
    </div>
  );
}

/* ══════════════════════════════════ */
export default function DetailPanel({ prediction, loading, history, onClose }: Props) {
  if (loading) return (
    <aside className="border border-line bg-surface flex items-center justify-center p-6 text-sm text-ink-muted animate-pulse-slow" style={rc}>
      Loading commune…
    </aside>
  );

  if (!prediction) return (
    <aside className="border border-line bg-surface flex items-center justify-center p-6 text-sm text-ink-muted text-center leading-6" style={rc}>
      Select a commune on the map<br />to view its analysis.
    </aside>
  );

  if ("prediction_available" in prediction) return (
    <aside className="border border-line bg-surface overflow-hidden flex flex-col" style={rc}>
      <Block>
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="font-semibold text-sm text-ink">{prediction.commune_name ?? prediction.commune_id}</p>
            <p className="text-xs text-ink-muted">{prediction.wilaya_name ?? "Algeria"}</p>
          </div>
          <button type="button" onClick={onClose} className="text-xs font-semibold text-ink-subtle hover:text-brand px-2 py-1 border border-line" style={rc}>Close</button>
        </div>
      </Block>
      <Div />
      <Block>
        <p className="text-xs font-semibold text-watch mb-2">No current prediction</p>
        <p className="text-sm text-ink-muted leading-6">{prediction.reason}</p>
        <p className="mt-3 text-xs text-ink-subtle">Note: "Unavailable" is not the same as LOW risk.</p>
      </Block>
    </aside>
  );

  const item = prediction as CommunePrediction;
  const priority = ["CRITICAL", "HIGH PRIORITY"].includes(normalizedUrgency(item));
  const style = RISK_STYLES[item.risk_label.toUpperCase()] ?? RISK_STYLES.LOW;
  const trust = reliabilityChip(item);

  return (
    <aside className="border border-line bg-surface overflow-hidden flex flex-col h-full min-h-0 animate-up" style={rc}>
        {/* ── header ── */}
      <Block>
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="font-semibold text-sm text-ink truncate">
              {item.commune_name}<span className="font-normal text-ink-muted"> · {item.wilaya_name}</span>
            </p>
            <p className="text-xs text-ink-subtle mt-0.5">Target: {formatDate(item.target_date)}</p>
          </div>
          <button type="button" onClick={onClose} className="shrink-0 text-xs font-semibold text-ink-subtle hover:text-brand px-2 py-1 border border-line" style={rc}>Close</button>
        </div>
      </Block>
      <Div />

      {/* ── risk badge ── */}
      <div className={`flex items-center justify-between px-4 py-3 ${style.bg} ${style.text}`}>
        <span className="font-data text-xl font-medium tracking-tight">{item.risk_label.toUpperCase()}</span>
        {priority && <span className="text-xs font-semibold opacity-90">{normalizedUrgency(item)}</span>}
      </div>

      {/* ── scrollable body ── */}
      <div className="overflow-y-auto flex-1 min-h-0">
        {/* probability breakdown */}
        <Block>
          <Label>Risk probability breakdown</Label>
          <ProbBar item={item} />
        </Block>
        <Div />

        {/* key numbers */}
        <div className="grid grid-cols-2" style={{ borderBottom: "1px solid var(--line)" }}>
          <div className="px-4 py-3" style={{ borderRight: "1px solid var(--line)" }}>
            <Label>P(HIGH)</Label>
            <p className="font-data text-2xl font-medium text-ink">{pct(item.prob_high)}</p>
            <p className="text-[11px] text-ink-subtle mt-0.5">Prob. of HIGH risk</p>
          </div>
          <div className="px-4 py-3">
            <Label>Urgency score</Label>
            <p className="font-data text-2xl font-medium text-ink">
              {Math.round(item.urgency_score)}<span className="text-sm font-normal text-ink-subtle">/100</span>
            </p>
            <p className="text-[11px] text-ink-subtle mt-0.5">{normalizedUrgency(item) || "Attention score"}</p>
          </div>
        </div>

        {/* action */}
        <Block>
          <Label>What should you do?</Label>
          <p className="text-sm font-semibold text-ink leading-snug">{recommendation(item)}</p>
        </Block>
        <Div />

        {/* SHAP drivers */}
        <Block>
          <Label>What is driving this prediction?</Label>
          <p className="text-xs text-ink-muted mb-3 leading-5">
            These are the factors the model weighed most heavily. Positive values pushed the prediction toward HIGH risk; negative values pushed it away.
          </p>
          <ShapDrivers item={item} />
        </Block>
        <Div />

        {/* current conditions */}
        <Block>
          <Label>Current conditions at this commune</Label>
          <CurrentConditions item={item} />
        </Block>
        <Div />

        {/* urgency context */}
        <Block>
          <Label>Why does urgency matter here?</Label>
          <p className="text-xs text-ink-muted leading-5 mb-3">
            The urgency score combines predicted risk with operational context: how many people live here, how reachable it is by road,
            and how much of the land can burn. A commune can be HIGH risk but low urgency (remote, unpopulated) — or MODERATE risk with
            very high urgency (dense population, poor road access). Urgency helps you prioritize when several communes are elevated.
          </p>
          <div className="grid grid-cols-2 gap-2">
            <MiniCard label="Population" value={item.pop_density_mean == null ? "—" : `${num(item.pop_density_mean, 0)}/km²`} note="People potentially exposed" />
            <MiniCard label="Road access" value={item.road_distance_mean_km == null ? "—" : `${num(item.road_distance_mean_km, 1)} km`} note="Avg distance to nearest road" />
          </div>
        </Block>
        <Div />

        {/* P(HIGH) history */}
        {history.length > 0 && (
          <>
            <Block>
              <Label>Recent P(HIGH) evolution — last {history.filter((p) => p.prob_high != null).length} days</Label>
              <Sparkline history={history} />
            </Block>
            <Div />
          </>
        )}

        {/* flags + reliability */}
        <Block>
          <div className="flex items-center justify-between mb-2.5">
            <Label>Prediction reliability & flags</Label>
            <span className={`text-[10px] font-semibold px-2 py-0.5 border ${trust.cls}`} style={rc}>{trust.label}</span>
          </div>
          <Flags item={item} />
        </Block>

      </div>
    </aside>
  );
}