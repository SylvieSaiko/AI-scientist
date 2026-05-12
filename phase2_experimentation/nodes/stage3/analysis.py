"""
Stage 3 template: Multivariable Cox PH + proportional hazards test + subgroup analysis.

Replace CSV, EVENT, TIME, EXPOSURE, and COVARIATES with your dataset's column names.
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
EXPOSURE = "exposure_col"                 # replace: primary exposure of interest

# Replace with covariate column names from your dataset
COVARIATES = [
    "covariate_1",
    "covariate_2",
    # add more...
]

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, proportional_hazard_test

    df = pd.read_csv(CSV, low_memory=False)
    for c in [EVENT, TIME, EXPOSURE] + COVARIATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fix censoring: non-events with TIME=0 should be censored at max follow-up
    # Adjust the horizon (365) to match your study's maximum follow-up in days
    # df.loc[(df[TIME] == 0) & (df[EVENT] == 0), TIME] = 365

    avail = [c for c in COVARIATES if c in df.columns]
    sdf   = df[[EVENT, TIME, EXPOSURE] + avail].dropna()
    sdf[TIME] = sdf[TIME].clip(lower=0.5)
    n_tot, n_ev = len(sdf), int(sdf[EVENT].sum())

    # ── Multivariable Cox ────────────────────────────────────────────────────
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(sdf, duration_col=TIME, event_col=EVENT)
    summary = cph.summary
    ci      = cph.concordance_index_
    print(f"Multivariable Cox: N={n_tot:,}, events={n_ev:,}, C-index={ci:.4f}")

    exp_r = summary.loc[EXPOSURE]
    metrics = {
        "n": n_tot, "events": n_ev, "c_index": round(ci, 4),
        "hr":      round(float(exp_r["exp(coef)"]), 4),
        "hr_lo":   round(float(exp_r["exp(coef) lower 95%"]), 4),
        "hr_hi":   round(float(exp_r["exp(coef) upper 95%"]), 4),
        "p_value": round(float(exp_r["p"]), 4),
    }

    # ── Proportional hazards test ────────────────────────────────────────────
    ph = proportional_hazard_test(cph, sdf, time_transform="rank")
    ph_fail = ph.summary[ph.summary["p"] < 0.05].index.tolist()
    metrics["ph_violations"] = ph_fail

    # ── KM + Cox forest plot ─────────────────────────────────────────────────
    s = summary[["exp(coef)", "exp(coef) lower 95%",
                  "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["HR", "lo", "hi", "p"]
    s = s.sort_values("HR")
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[1]
    colors = ["tomato" if p < 0.05 else "steelblue" for p in s["p"]]
    ax.barh(y, s["HR"] - 1, left=1, height=0.55, color=colors, alpha=0.8)
    ax.errorbar(s["HR"].values, y,
                xerr=[s["HR"].values - s["lo"].values,
                       s["hi"].values - s["HR"].values],
                fmt="none", color="black", capsize=3)
    ax.axvline(1, color="black", linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(s.index, fontsize=10)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title("Multivariable Cox — Primary Outcome")

    ax2 = axes[0]
    for val, lbl, col in [(1, "Exposed", "tomato"), (0, "Unexposed", "steelblue")]:
        grp = sdf[sdf[EXPOSURE] == val]
        km  = KaplanMeierFitter()
        km.fit(grp[TIME], grp[EVENT], label=f"{lbl} (n={len(grp):,})")
        km.plot_survival_function(ax=ax2, ci_show=True, color=col)
    lr = logrank_test(sdf[sdf[EXPOSURE] == 0][TIME], sdf[sdf[EXPOSURE] == 1][TIME],
                      sdf[sdf[EXPOSURE] == 0][EVENT], sdf[sdf[EXPOSURE] == 1][EVENT])
    ax2.set_title("Event-Free Survival by Exposure")
    ax2.set_xlabel("Time"); ax2.set_ylabel("Event-free probability")
    ax2.text(0.05, 0.05, f"Log-rank p={lr.p_value:.2e}", transform=ax2.transAxes, fontsize=10)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(NODE_DIR, "figures", "multivariable_cox.png"), dpi=150)
    plt.close()

    # ── Subgroup analysis ────────────────────────────────────────────────────
    subgroups = {}

    def cox_exposure_subgroup(mask, label):
        sub = sdf[mask][[EVENT, TIME, EXPOSURE]].dropna()
        sub[TIME] = sub[TIME].clip(lower=0.5)
        if len(sub) < 50 or sub[EVENT].sum() < 10:
            return
        try:
            c = CoxPHFitter(penalizer=0.1)
            c.fit(sub, duration_col=TIME, event_col=EVENT)
            r = c.summary.loc[EXPOSURE]
            subgroups[label] = {
                "HR": round(float(r["exp(coef)"]), 3),
                "lo": round(float(r["exp(coef) lower 95%"]), 3),
                "hi": round(float(r["exp(coef) upper 95%"]), 3),
                "p":  round(float(r["p"]), 4),
                "n":  len(sub), "events": int(sub[EVENT].sum()),
            }
        except Exception:
            pass

    # Add your pre-specified subgroup masks here, e.g.:
    # cox_exposure_subgroup(df["age"] >= 65, "Age ≥65")
    # cox_exposure_subgroup(df["age"] < 65,  "Age <65")

    metrics["subgroups"] = subgroups

    with open(os.path.join(NODE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(NODE_DIR, "results.txt"), "w") as f:
        f.write(f"N={n_tot:,}, events={n_ev:,}, C-index={ci:.4f}\n")
        f.write(f"Exposure HR={metrics['hr']:.3f} "
                f"[{metrics['hr_lo']:.3f}-{metrics['hr_hi']:.3f}] "
                f"p={metrics['p_value']:.4f}\n")
        f.write(f"PH violations: {ph_fail or 'None'}\n")
    print("\nSTATUS: SUCCESS")
except Exception:
    traceback.print_exc(); print("STATUS: FAILED")
