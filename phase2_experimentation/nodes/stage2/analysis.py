"""
Stage 2 template: LASSO-penalised Cox regression for variable selection.

Replace CSV, EVENT, TIME, and CANDIDATES with your dataset's column names.
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

# Replace with candidate predictor column names from your dataset
CANDIDATES = [
    "predictor_1",
    "predictor_2",
    "predictor_3",
    # add more...
]

try:
    from lifelines import CoxPHFitter

    df = pd.read_csv(CSV, low_memory=False)
    for c in [EVENT, TIME] + CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    avail = [c for c in CANDIDATES if c in df.columns]
    sdf   = df[[EVENT, TIME] + avail].dropna()
    # Fix censoring if non-events have TIME=0
    # sdf.loc[(sdf[TIME] == 0) & (sdf[EVENT] == 0), TIME] = 365
    sdf[TIME] = sdf[TIME].clip(lower=0.5)

    print(f"LASSO cohort: n={len(sdf):,}, events={int(sdf[EVENT].sum()):,}")

    best_lambda, best_ci = 0.1, 0.0
    for lam in [0.001, 0.01, 0.05, 0.1, 0.5]:
        try:
            c = CoxPHFitter(penalizer=lam, l1_ratio=1.0)
            c.fit(sdf, duration_col=TIME, event_col=EVENT)
            ci = c.concordance_index_
            if ci > best_ci:
                best_ci, best_lambda = ci, lam
        except Exception:
            pass

    cph = CoxPHFitter(penalizer=best_lambda, l1_ratio=1.0)
    cph.fit(sdf, duration_col=TIME, event_col=EVENT)
    summary  = cph.summary
    selected = summary[summary["exp(coef)"].round(4) != 1.0].index.tolist()
    print(f"Lambda={best_lambda}, C-index={best_ci:.4f}")
    print("Selected predictors:", selected)

    # Forest plot
    if selected:
        s = summary.loc[selected, ["exp(coef)", "exp(coef) lower 95%",
                                    "exp(coef) upper 95%", "p"]]
        s.columns = ["HR", "lo", "hi", "p"]
        s = s.sort_values("HR")
        y = np.arange(len(s))
        fig, ax = plt.subplots(figsize=(9, max(4, len(s) * 0.6)))
        colors = ["tomato" if p < 0.05 else "steelblue" for p in s["p"]]
        ax.barh(y, s["HR"] - 1, left=1, height=0.55, color=colors, alpha=0.8)
        ax.errorbar(s["HR"].values, y,
                    xerr=[s["HR"].values - s["lo"].values,
                           s["hi"].values - s["HR"].values],
                    fmt="none", color="black", capsize=3)
        ax.axvline(1, color="black", linestyle="--")
        ax.set_yticks(y); ax.set_yticklabels(s.index, fontsize=10)
        ax.set_xlabel("Hazard Ratio (95% CI)")
        ax.set_title(f"LASSO Cox (λ={best_lambda}) — Selected Predictors")
        plt.tight_layout()
        fig.savefig(os.path.join(NODE_DIR, "figures", "lasso_forest.png"), dpi=150)
        plt.close()

    metrics = {
        "n": int(len(sdf)), "events": int(sdf[EVENT].sum()),
        "c_index": round(best_ci, 4), "best_lambda": best_lambda,
        "selected_predictors": selected,
    }
    with open(os.path.join(NODE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(NODE_DIR, "results.txt"), "w") as f:
        f.write(f"LASSO Cox  λ={best_lambda}  C-index={best_ci:.4f}\n")
        f.write(f"N={len(sdf):,}  Events={int(sdf[EVENT].sum()):,}\n")
        f.write(f"Selected: {selected}\n\n")
        f.write(summary[["exp(coef)", "exp(coef) lower 95%",
                          "exp(coef) upper 95%", "p"]].to_string())
    print("STATUS: SUCCESS")
except Exception:
    traceback.print_exc(); print("STATUS: FAILED")
