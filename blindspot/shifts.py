"""Controlled corruptions applied to TEST data only. Statistics come from clean training data."""
import numpy as np, pandas as pd

KINDS = ("noise", "missing", "outliers")
SEVERITIES = (0.2, 0.5, 1.0)

def feature_stats(Xtr, num_cols):
    return Xtr[num_cols].mean(), Xtr[num_cols].std().replace(0, 1.0)

def corrupt(X, kind, severity, rng, num_cols, cat_cols, stats):
    """noise:    add N(0, (severity*std)^2) to numeric cells
       missing:  blank a fraction 0.4*severity of ALL cells (numeric and categorical)
       outliers: replace a fraction 0.1*severity of numeric cells with mean +/- 6 std"""
    X = X.copy()                                        # never mutate the caller's data
    mean, std = stats
    if kind == "noise":
        for c in num_cols:
            X[c] = X[c] + rng.normal(0, severity * std[c], len(X))
    elif kind == "missing":
        for c in list(num_cols) + list(cat_cols):
            col = X[c].astype(float) if c in num_cols else X[c].astype(object)
            col[rng.random(len(X)) < 0.4 * severity] = np.nan
            X[c] = col
    elif kind == "outliers":
        for c in num_cols:
            m = rng.random(len(X)) < 0.1 * severity
            X[c] = X[c].astype(float)
            X.loc[m, c] = mean[c] + rng.choice([-6, 6], m.sum()) * std[c]
    else:
        raise ValueError(kind)
    return X
