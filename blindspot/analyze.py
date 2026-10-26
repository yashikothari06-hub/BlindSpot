"""Summaries, paired statistics and figures.  python -m blindspot.analyze results/results.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

METRICS = ["accuracy", "macro_f1", "error_rate", "ece_raw", "ece_sigmoid", "ece_isotonic", "auroc", "auprc",
           "recall_p50", "recall_top20", "risk80", "aurc"]
KEY = ["dataset", "model", "shift_type", "severity", "predictor"]

def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x.dropna()); rng = np.random.default_rng(seed)
    if len(x) < 2: return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(1); return tuple(np.percentile(m, [2.5, 97.5]))

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p, dtype=float); run = 0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1, run)
    return adj

def main(path):
    df = pd.read_csv(path); out = Path(path).parent
    df["auprc_lift"] = df["auprc"] / df["error_rate"]       # AUPRC relative to the error-rate baseline
    g = df.groupby(KEY)
    summ = g[METRICS + ["auprc_lift"]].mean()
    for m in ["auprc", "auroc", "accuracy"]:
        ci = g[m].apply(boot_ci); summ[m + "_ci_lo"] = ci.map(lambda t: t[0]); summ[m + "_ci_hi"] = ci.map(lambda t: t[1])
    summ["n_seeds"] = g.size(); summ.reset_index().to_csv(out / "summary.csv", index=False)

    # paired Wilcoxon on AUPRC across seeds: each predictor vs confidence baseline, Holm-corrected
    rows = []
    for (d, m, k, s), sub in df.groupby(["dataset", "model", "shift_type", "severity"]):
        piv = sub.pivot(index="seed", columns="predictor", values="auprc")
        for p in ["pca_recon", "tree", "gbm"]:
            diff = (piv[p] - piv["confidence"]).dropna()
            pv = wilcoxon(diff).pvalue if len(diff) >= 5 and (diff != 0).any() else np.nan
            rows.append(dict(dataset=d, model=m, shift_type=k, severity=s, predictor=p,
                             mean_diff_auprc=diff.mean(), n=len(diff), p_raw=pv))
    st = pd.DataFrame(rows); ok = st.p_raw.notna()
    st.loc[ok, "p_holm"] = holm(st.loc[ok, "p_raw"].values)
    st["significant_0.05"] = st.p_holm < 0.05; st.to_csv(out / "stats_vs_confidence.csv", index=False)

    # Figure 1: accuracy and calibration degrade with corruption severity
    base = df[df.predictor == "confidence"]; ds = sorted(base.dataset.unique())
    fig, ax = plt.subplots(len(ds), 3, figsize=(13, 3.4 * len(ds)), squeeze=False)
    for i, d in enumerate(ds):
        for j, kind in enumerate(["noise", "missing", "outliers"]):
            for m, sub in base[base.dataset == d].groupby("model"):
                clean = sub[sub.shift_type == "clean"].accuracy.mean()
                c = sub[sub.shift_type == kind].groupby("severity").accuracy.mean()
                ax[i, j].plot([0] + list(c.index), [clean] + list(c.values), marker="o", label=m)
            ax[i, j].set_title(f"{d}: {kind}"); ax[i, j].set_xlabel("severity"); ax[i, j].set_ylabel("accuracy")
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(out / "fig_degradation.png", dpi=130); plt.close(fig)

    # Figure 2: AUPRC lift of each failure predictor, clean vs covariate shift
    sub = df[df.shift_type.isin(["clean", "covariate_shift"])]
    t = sub.groupby(["dataset", "shift_type", "predictor"]).auprc_lift.mean().unstack("predictor")
    fig, ax = plt.subplots(figsize=(11, 4))
    t.plot.bar(ax=ax)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_ylabel("AUPRC / error rate")
    ax.set_title("Failure-predictor AUPRC relative to the error-rate baseline (1.0 = chance)")
    fig.tight_layout()
    plot_path = (out.resolve() / "fig_predictor_lift.png")
    fig.savefig(str(plot_path), format="png", dpi=130)
    plt.close(fig)
    print("Saved predictor-lift figure to:", plot_path)

    print("wrote summary.csv, stats_vs_confidence.csv, fig_degradation.png, fig_predictor_lift.png in", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/results.csv")
