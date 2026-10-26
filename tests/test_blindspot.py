import numpy as np, pandas as pd
from blindspot.data import load
from blindspot.metrics import ece, failure_metrics
from blindspot.shifts import corrupt, feature_stats
from blindspot.run import run_one

def test_corrupt_does_not_mutate_and_only_touches_test_data():
    X, y, _ = load("synthetic", 1000); num = list(X.select_dtypes("number").columns)
    cat = [c for c in X.columns if c not in num]; before = X.copy()
    for kind in ("noise", "missing", "outliers"):
        Xc = corrupt(X, kind, 1.0, np.random.default_rng(0), num, cat, feature_stats(X, num))
        assert not Xc.equals(X)
    pd.testing.assert_frame_equal(X, before)

def test_missing_severity_matches_target_fraction():
    X, _, _ = load("synthetic", 3000); num = list(X.select_dtypes("number").columns)
    cat = [c for c in X.columns if c not in num]
    Xc = corrupt(X, "missing", 0.5, np.random.default_rng(1), num, cat, feature_stats(X, num))
    assert abs(Xc.isna().to_numpy().mean() - 0.2) < 0.02

def test_failure_metrics_perfect_and_random():
    err = np.array([0] * 80 + [1] * 20)
    assert failure_metrics(err.astype(float), err)["auroc"] == 1.0
    rnd = np.random.default_rng(0).random(5000); e = (np.random.default_rng(1).random(5000) < 0.2).astype(int)
    assert abs(failure_metrics(rnd, e)["auroc"] - 0.5) < 0.05

def test_ece_zero_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(0); conf = rng.uniform(0.5, 1, 20000)
    assert ece(conf, rng.random(20000) < conf) < 0.02
    assert ece(np.full(1000, 0.99), np.arange(1000) % 2 == 0) > 0.4

def test_pipeline_produces_all_conditions_and_predictors():
    X, y, sh = load("synthetic", 1500)
    rows = pd.DataFrame(run_one(X, y, sh, "logreg", 0, "synthetic"))
    assert set(rows.shift_type) == {"clean", "noise", "missing", "outliers", "covariate_shift"}
    assert set(rows.predictor) == {"confidence", "pca_recon", "tree", "gbm"}
    assert rows.accuracy.between(0, 1).all()
