import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def ece(conf, correct, bins=10):
    """Expected calibration error: weighted gap between confidence and accuracy per bin."""
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))

def failure_metrics(score, err):
    """score: higher = more likely wrong. err: 1 if the base model was wrong."""
    err = err.astype(int); base = err.mean()
    if err.min() == err.max():
        return dict(auroc=np.nan, auprc=np.nan, recall_p50=np.nan, recall_top20=np.nan,
                    risk80=base, aurc=base)
    p, r, _ = precision_recall_curve(err, score)
    order = np.argsort(score, kind="stable")            # lowest failure score = most trusted
    risk_curve = np.cumsum(err[order]) / np.arange(1, len(err) + 1)   # risk vs coverage
    top = np.argsort(-score, kind="stable")[: max(1, int(0.2 * len(err)))]
    return dict(auroc=roc_auc_score(err, score), auprc=average_precision_score(err, score),
                recall_p50=float(r[p >= 0.5].max()) if (p >= 0.5).any() else 0.0,
                recall_top20=err[top].sum() / err.sum(),
                risk80=float(risk_curve[int(0.8 * len(err)) - 1]), aurc=float(risk_curve.mean()))
