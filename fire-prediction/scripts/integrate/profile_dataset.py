"""
Generates the "Final dataset — descriptive statistics" section for the
technical documentation. Run after build_dataset.py has produced
data/integrated/algeria_wildfire_dataset.parquet.
"""

import pandas as pd
from pathlib import Path

DATASET_PATH = Path("data/integrated/algeria_wildfire_dataset.parquet")

RISK_LABELS = {0: "NO_FIRE", 1: "LOW", 2: "MODERATE", 3: "HIGH", 4: "CRITICAL"}


def main():
    df = pd.read_parquet(DATASET_PATH)

    print("=" * 70)
    print("FINAL DATASET - DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # ── Overview ─────────────────────────────────────────────────────────────
    print(f"\n## Overview\n")
    print(f"- Rows: {len(df):,}")
    print(f"- Columns: {df.shape[1]}")
    print(f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"- Wilayas covered: {df['wilaya_id'].nunique()} / 48")
    print(f"- Years spanned: {sorted(df['year'].unique())}")
    print(f"- File size: {DATASET_PATH.stat().st_size / 1e6:.1f} MB")

    # ── Rows per year (checks for gaps) ─────────────────────────────────────
    print(f"\n## Rows per year\n")
    print(f"| Year | Rows | Fire-season days x wilayas expected |")
    print(f"|---|---|---|")
    for year, count in df.groupby("year").size().items():
        print(f"| {year} | {count:,} | |")

    # ── Target class distribution ───────────────────────────────────────────
    print(f"\n## Target class distribution (fire_risk_class)\n")
    print(f"| Class | Label | Count | % of rows |")
    print(f"|---|---|---|---|")
    dist = df["fire_risk_class"].value_counts().sort_index()
    for cls, count in dist.items():
        pct = 100 * count / len(df)
        label = RISK_LABELS.get(cls, "?")
        print(f"| {cls} | {label} | {count:,} | {pct:.2f}% |")

    fire_pct = 100 * (df["fire_risk_class"] > 0).mean()
    print(f"\n- Fire days (any class > 0): {fire_pct:.2f}% of rows")
    print(f"- Class imbalance ratio (NO_FIRE : rest): "
          f"{dist.get(0,0) / max(len(df) - dist.get(0,0), 1):.1f} : 1")

    # ── Missing values per column ───────────────────────────────────────────
    print(f"\n## Missing values\n")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("None.")
    else:
        print(f"| Column | Missing | % |")
        print(f"|---|---|---|")
        for col, n in missing.items():
            print(f"| {col} | {n:,} | {100*n/len(df):.2f}% |")

    # ── Numeric feature summary stats ───────────────────────────────────────
    print(f"\n## Numeric feature summary\n")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    exclude = ["year", "day_of_year", "wilaya_id", "era5_cell_id", "fire_risk_class"]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    summary = df[numeric_cols].describe().T[["min", "mean", "50%", "max", "std"]]
    summary.columns = ["min", "mean", "median", "max", "std"]
    print(f"| Feature | Min | Mean | Median | Max | Std |")
    print(f"|---|---|---|---|---|---|")
    for col, row in summary.iterrows():
        print(f"| {col} | {row['min']:.2f} | {row['mean']:.2f} | "
              f"{row['median']:.2f} | {row['max']:.2f} | {row['std']:.2f} |")

    # ── Top / bottom wilayas by fire activity ───────────────────────────────
    print(f"\n## Wilayas by total fire detections (top 5 / bottom 5)\n")
    by_wilaya = (df.groupby("wilaya_name")["fire_count"]
                 .sum().sort_values(ascending=False))
    print(f"Top 5:")
    for name, count in by_wilaya.head(5).items():
        print(f"  {name}: {int(count):,}")
    print(f"Bottom 5:")
    for name, count in by_wilaya.tail(5).items():
        print(f"  {name}: {int(count):,}")

    # ── FWI sanity check: fire season should show a clear seasonal curve ────
    print(f"\n## Mean FWI by month (seasonal sanity check)\n")
    print(f"| Month | Mean FWI | Mean fire_count |")
    print(f"|---|---|---|")
    by_month = df.groupby("month")[["FWI", "fire_count"]].mean()
    for month, row in by_month.iterrows():
        print(f"| {month} | {row['FWI']:.1f} | {row['fire_count']:.3f} |")

if __name__ == "__main__":
    main()