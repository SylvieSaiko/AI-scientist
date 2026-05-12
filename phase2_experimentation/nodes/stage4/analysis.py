"""
Stage 4 template: Temporal trends, era-stratified Cox, and competing risks context.

Replace CSV, EVENT, TIME, EXPOSURE, and DATE_COL with your dataset's column names.
"""
import os, json, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV      = "/path/to/your/dataset.csv"   # replace
EVENT    = "event_col"                    # replace
TIME     = "time_col"                     # replace
EXPOSURE = "exposure_col"                 # replace: primary exposure (e.g. binary 0/1)
DATE_COL = "date_col"                     # replace: procedure/index date column

# Replace with covariate column names for era-stratified Cox
COVARIATES = [
    "covariate_1",
    "covariate_2",
    # add more...
]

# Adjust era cut points and labels to match your study period
ERA_BINS   = [1999, 2009, 2014, 2026]    # replace with your year boundaries
ERA_LABELS = ["2000–2009", "2010–2014", "2015–2025"]  # replace

# Set the follow-up horizon for the endpoint (in days) so non-events can be
# censored at the protocol maximum rather than left at 0.
HORIZON = 365    # replace: maximum follow-up in days

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter

    df = pd.read_csv(CSV, low_memory=False)
    for c in [EVENT, TIME, EXPOSURE] + COVARIATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fix censoring: registry records time only for event patients;
    # non-events were followed to HORIZON days → censor there
    # df.loc[(df[TIME] == 0) & (df[EVENT] == 0), TIME] = HORIZON

    # ── Derive year from date column ─────────────────────────────────────────
    df["_year"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.year
    # Fallback: try first 4 characters of the string
    mask = df["_year"].isna() & df[DATE_COL].notna()
    df.loc[mask, "_year"] = (
        df.loc[mask, DATE_COL].astype(str).str[:4]
        .pipe(pd.to_numeric, errors="coerce")
    )

    # ── Annual trends: exposure prevalence + annual event rate ───────────────
    sdf = df[[EVENT, TIME, EXPOSURE, "_year"]].dropna()
    sdf[TIME] = sdf[TIME].clip(lower=0.5)

    yr_range = sorted(y for y in sdf["_year"].dropna().unique()
                      if ERA_BINS[0] < y <= ERA_BINS[-1])

    trend_rows = []
    for yr in yr_range:
        y_df = sdf[sdf["_year"] == yr]
        if len(y_df) < 20:
            continue
        exp_prev  = (y_df[EXPOSURE] == 1).mean() * 100
        rate_exp  = y_df[y_df[EXPOSURE] == 1][EVENT].mean() * 100 if (y_df[EXPOSURE] == 1).sum() > 5 else np.nan
        rate_unex = y_df[y_df[EXPOSURE] == 0][EVENT].mean() * 100 if (y_df[EXPOSURE] == 0).sum() > 5 else np.nan
        trend_rows.append({
            "year":       int(yr),
            "n":          len(y_df),
            "exp_prev":   round(exp_prev, 1),
            "rate_exp":   round(rate_exp, 1)   if pd.notna(rate_exp)  else None,
            "rate_unexp": round(rate_unex, 1)  if pd.notna(rate_unex) else None,
        })

    tdf = pd.DataFrame(trend_rows) if trend_rows else pd.DataFrame()
    if not tdf.empty:
        tdf = tdf.dropna(subset=["rate_exp", "rate_unexp"])

    if len(tdf) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.plot(tdf["year"], tdf["rate_exp"],   "o-", color="tomato",    label="Exposed",    linewidth=2)
        ax1.plot(tdf["year"], tdf["rate_unexp"],  "s-", color="steelblue", label="Unexposed",  linewidth=2)
        ax1.set_title(f"Annual Event Rate by Exposure Status", fontsize=11)
        ax1.set_xlabel("Year"); ax1.set_ylabel("Event rate (%)")
        ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.bar(tdf["year"], tdf["exp_prev"], color="steelblue", alpha=0.7)
        ax2.set_title("Annual Exposure Prevalence", fontsize=11)
        ax2.set_xlabel("Year"); ax2.set_ylabel("Prevalence (%)")
        ax2.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(os.path.join(NODE_DIR, "figures", "temporal_trends.png"), dpi=150)
        plt.close()

    # ── Era-stratified Cox ────────────────────────────────────────────────────
    era_results = {}
    df["_era"] = pd.cut(df["_year"], bins=ERA_BINS, labels=ERA_LABELS)
    avail = [c for c in COVARIATES if c in df.columns]

    for era in ERA_LABELS:
        tmp = df[df["_era"] == era][[EVENT, TIME, EXPOSURE] + avail].dropna()
        tmp[TIME] = tmp[TIME].clip(lower=0.5)
        if len(tmp) < 50 or tmp[EVENT].sum() < 10:
            continue
        try:
            c = CoxPHFitter(penalizer=0.1)
            c.fit(tmp, duration_col=TIME, event_col=EVENT)
            r = c.summary.loc[EXPOSURE]
            era_results[era] = {
                "n":      len(tmp),
                "events": int(tmp[EVENT].sum()),
                "HR":     round(float(r["exp(coef)"]), 3),
                "lo":     round(float(r["exp(coef) lower 95%"]), 3),
                "hi":     round(float(r["exp(coef) upper 95%"]), 3),
                "p":      round(float(r["p"]), 4),
            }
        except Exception:
            pass

    if era_results:
        labs   = list(era_results.keys())
        hrs    = [era_results[l]["HR"] for l in labs]
        los    = [era_results[l]["lo"] for l in labs]
        his    = [era_results[l]["hi"] for l in labs]
        ps     = [era_results[l]["p"]  for l in labs]
        y      = np.arange(len(labs))
        fig, ax = plt.subplots(figsize=(9, 4))
        colors  = ["tomato" if p < 0.05 else "steelblue" for p in ps]
        ax.barh(y, np.array(hrs) - 1, left=1, height=0.55, color=colors, alpha=0.8)
        ax.errorbar(hrs, y,
                    xerr=[np.array(hrs) - np.array(los),
                           np.array(his) - np.array(hrs)],
                    fmt="none", color="black", capsize=3)
        ax.axvline(1, color="black", linestyle="--")
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=11)
        ax.set_xlabel("Adjusted HR for Exposure (95% CI)")
        ax.set_title("Era-Stratified Cox Analysis", fontsize=12)
        for i, (hr, lo, hi, p, l) in enumerate(zip(hrs, los, his, ps, labs)):
            n = era_results[l]["n"]
            ax.text(max(his) + 0.05, i,
                    f"HR={hr:.2f} [{lo:.2f}–{hi:.2f}]  p={p:.3f}  n={n:,}",
                    va="center", fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.72, 1])
        fig.savefig(os.path.join(NODE_DIR, "figures", "era_forest.png"), dpi=150)
        plt.close()

    # ── Competing risks context ───────────────────────────────────────────────
    # Replace COMPETING_EVENT with your competing event column (e.g. all-cause death)
    # if it is available in the dataset.
    COMPETING_EVENT = "competing_event_col"   # replace or remove this block
    cr_summary = {}
    if COMPETING_EVENT in df.columns:
        cr_df = df[[EVENT, TIME, EXPOSURE, COMPETING_EVENT]].dropna(subset=[EVENT, TIME])
        cr_df[TIME] = cr_df[TIME].clip(lower=0.5)
        cr_df[COMPETING_EVENT] = pd.to_numeric(cr_df[COMPETING_EVENT], errors="coerce").fillna(0)
        cr_df["cr"] = 0
        cr_df.loc[cr_df[EVENT] == 1, "cr"] = 1
        cr_df.loc[(cr_df[COMPETING_EVENT] == 1) & (cr_df[EVENT] == 0), "cr"] = 2
        cr_summary = {
            "primary_events":      int((cr_df["cr"] == 1).sum()),
            "competing_no_event":  int((cr_df["cr"] == 2).sum()),
            "censored":            int((cr_df["cr"] == 0).sum()),
            "rate_exp":   round(cr_df[cr_df[EXPOSURE] == 1]["cr"].eq(1).mean() * 100, 2),
            "rate_unexp": round(cr_df[cr_df[EXPOSURE] == 0]["cr"].eq(1).mean() * 100, 2),
        }

    metrics = {
        "trend_data":      trend_rows,
        "era_results":     era_results,
        "competing_risks": cr_summary,
        "n_with_year":     int(sdf["_year"].notna().sum()),
    }
    with open(os.path.join(NODE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(NODE_DIR, "results.txt"), "w", encoding="utf-8") as f:
        f.write("=== Temporal Trends ===\n")
        if trend_rows:
            f.write(pd.DataFrame(trend_rows).to_string(index=False))
        f.write("\n\n=== Era-Stratified Cox ===\n")
        for k, v in era_results.items():
            f.write(f"  {k}: HR={v['HR']:.3f} [{v['lo']:.3f}–{v['hi']:.3f}] "
                    f"p={v['p']:.4f} n={v['n']:,}\n")
        f.write(f"\n=== Competing Risks ===\n{json.dumps(cr_summary, indent=2)}\n")

    print("STATUS: SUCCESS")
except Exception:
    traceback.print_exc(); print("STATUS: FAILED")
