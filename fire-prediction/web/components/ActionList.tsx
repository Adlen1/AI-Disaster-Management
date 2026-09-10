"use client";

import type { CommunePrediction } from "../lib/types";

type Props = {
  items: CommunePrediction[];
  selectedId: string | null;
  onSelect: (item: CommunePrediction) => void;
  priorityOnly?: boolean;
};

const RISK_GROUPS = [
  { key: "HIGH",     label: "High risk",     barClass: "risk-bar-high", dotClass: "bg-danger", textClass: "text-danger" },
  { key: "MODERATE", label: "Moderate risk", barClass: "risk-bar-mod",  dotClass: "bg-watch",  textClass: "text-watch"  },
  { key: "LOW",      label: "Low risk",      barClass: "risk-bar-low",  dotClass: "bg-safe",   textClass: "text-safe"   },
] as const;

export function isPriority(item: CommunePrediction) {
  const label = (item.urgency_label ?? "").trim().toUpperCase().replaceAll("_", " ");
  return label === "CRITICAL" || label === "HIGH PRIORITY";
}

export function reasonCategory(item: CommunePrediction) {
  const feature = item.top_drivers?.[0]?.feature ?? "";
  const labels: Record<string, string> = {
    fire_count: "Recent fire activity",
    fire_count_3d: "Recent fire activity",
    fire_count_7d: "Recent fire activity",
    frp_total: "Recent fire intensity",
    FWI: "Hot and dry conditions",
    temp_c: "Hot and dry conditions",
    rh: "Hot and dry conditions",
    DMC: "Dry vegetation",
    DC: "Dry vegetation",
    NDVI: "Dry vegetation",
    NBR: "Dry vegetation",
    burnable_fraction: "Dry vegetation",
    pop_density_mean: "Population exposure",
    road_distance_mean_km: "Limited road access",
  };
  return labels[feature] ?? "Several fire, weather, and landscape signals";
}

function CommuneRow({
  item,
  selectedId,
  onSelect,
}: {
  item: CommunePrediction;
  selectedId: string | null;
  onSelect: (item: CommunePrediction) => void;
}) {
  const selected = selectedId === item.commune_id;
  const priority = isPriority(item);
  const barClass =
    item.risk_label.toUpperCase() === "HIGH"
      ? "risk-bar-high"
      : item.risk_label.toUpperCase() === "MODERATE"
      ? "risk-bar-mod"
      : "risk-bar-low";

  return (
    <button
      type="button"
      onClick={() => onSelect(item)}
      className={[
        "w-full text-left px-4 py-3 flex items-center justify-between gap-4",
        "border-b border-line last:border-b-0 row-hover",
        "focus:outline-none focus:ring-2 focus:ring-inset focus:ring-brand/30",
        barClass,
        selected
          ? "bg-brand/8 text-ink"
          : "bg-surface hover:bg-surface-muted text-ink",
      ].join(" ")}
      aria-pressed={selected}
    >
      {/* name + wilaya */}
      <span className="min-w-0 flex-1">
        <span className="block truncate font-semibold text-sm text-ink leading-snug">
          {item.commune_name}
          <span className="font-normal text-ink-muted"> · {item.wilaya_name}</span>
        </span>
        <span className="block mt-0.5 text-xs text-ink-subtle">
          {reasonCategory(item)}
        </span>
      </span>

      {/* urgency score + priority flag */}
      <span className="shrink-0 flex flex-col items-end gap-0.5">
        <span className="font-data text-base font-medium text-ink leading-none">
          {Math.round(item.urgency_score)}
          <span className="text-xs text-ink-subtle font-normal">/100</span>
        </span>
        {priority && (
          <span className="text-[10px] font-semibold text-brand tracking-wide">
            Priority
          </span>
        )}
      </span>
    </button>
  );
}

function GroupHeader({
  group,
  count,
}: {
  group: (typeof RISK_GROUPS)[number];
  count: number;
}) {
  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-line bg-canvas sticky top-0 z-10">
      <span className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full shrink-0 ${group.dotClass}`} aria-hidden="true" />
        <span className={`text-xs font-semibold ${group.textClass}`}>{group.label}</span>
      </span>
      <span className="font-data text-sm font-medium text-ink">{count}</span>
    </div>
  );
}

export default function ActionList({ items, selectedId, onSelect, priorityOnly = false }: Props) {
  /* ── Priority-only mode ── */
  if (priorityOnly) {
    const priorityItems = items
      .filter(isPriority)
      .sort((a, b) => b.urgency_score - a.urgency_score);

    return (
      <div className="border border-brand/40 bg-surface overflow-hidden" style={{ borderRadius: "var(--radius-control)" }}>
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-brand/30 bg-brand/5">
          <span className="text-xs font-semibold text-brand">Priority communes</span>
          <span className="font-data text-sm font-medium text-ink">{priorityItems.length}</span>
        </div>
        <div className="max-h-[32rem] overflow-y-auto">
          {priorityItems.length ? (
            priorityItems.map((item) => (
              <CommuneRow key={item.commune_id} item={item} selectedId={selectedId} onSelect={onSelect} />
            ))
          ) : (
            <p className="px-4 py-5 text-sm text-ink-muted">No priority communes in this filter.</p>
          )}
        </div>
      </div>
    );
  }

  /* ── Full grouped list ── */
  return (
    <div className="border border-line bg-surface overflow-hidden" style={{ borderRadius: "var(--radius-control)" }}>
      {RISK_GROUPS.map((group) => {
        const rows = items
          .filter((item) => item.risk_label.toUpperCase() === group.key)
          .sort((a, b) => b.urgency_score - a.urgency_score);

        return (
          <section key={group.key}>
            <GroupHeader group={group} count={rows.length} />
            <div className="max-h-56 overflow-y-auto">
              {rows.length ? (
                rows.map((item) => (
                  <CommuneRow key={item.commune_id} item={item} selectedId={selectedId} onSelect={onSelect} />
                ))
              ) : (
                <p className="px-4 py-3 text-xs text-ink-subtle">No communes in this group.</p>
              )}
            </div>
          </section>
        );
      })}
    </div>
  );
}