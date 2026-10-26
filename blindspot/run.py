"""Main experiment runner.  python -m blindspot.run --datasets adult bank wine --seeds 10"""
import argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from .data import load
from .metrics import ece, failure_metrics
from .shifts import KINDS, SEVERITIES, corrupt, feature_stats

warnings.filterwarnings("ignore")

def make_pre(num, cat):
    return ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])

def make_clf(name, seed):
    return {"logreg": lambda: LogisticRegression(max_iter=1000),
            "svm": lambda: SVC(probability=True, random_state=seed),
            "rf": lambda: RandomForestClassifier(200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
            "gbm": lambda: HistGradientBoostingClassifier(random_state=seed)}[name]()

def run_one(X, y, shifted, model_name, seed, name="data"):
    num = list(X.select_dtypes("number").columns); cat = [c for c in X.columns if c not in num]
        
    if shifted is None:
        # Custom datasets: split all rows into training and testing sets.
        indices = np.arange(len(y))
        tr, te = train_test_split(
            indices,
            test_size=0.3,
            stratify=y,
            random_state=seed
        )
        Xsh, ysh = None, None
    else:
        # Existing datasets: preserve their predefined shifted groups.
        tr, te = train_test_split(
            np.where(~shifted)[0],
            test_size=0.3,
            stratify=y[~shifted],
            random_state=seed
        )
        sh = np.where(shifted)[0]
        Xsh, ysh = X.iloc[sh], y[sh]

    Xtr, ytr = X.iloc[tr], y[tr]
    Xte, yte = X.iloc[te], y[te]


    # 1) base model, fit on clean training data only (preprocessing is inside the pipeline)
    model = Pipeline([("pre", make_pre(num, cat)), ("clf", make_clf(model_name, seed))])

    # 2) out-of-fold probabilities: honest errors for training the failure predictors
    oof = cross_val_predict(clone(model), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                            method="predict_proba")
    oof_conf, oof_err = oof.max(1), (oof.argmax(1) != ytr).astype(int)
    model.fit(Xtr, ytr)

    # 3) PCA on clean training features -> reconstruction distance
    pre = model.named_steps["pre"]; Ztr = pre.transform(Xtr)
    pca = PCA(n_components=0.90, svd_solver="full").fit(Ztr)
    recon = lambda Z: np.linalg.norm(Z - pca.inverse_transform(pca.transform(Z)), axis=1)

    # 4) failure predictors trained on OOF error labels + OOF confidence
    feats = lambda Z, conf: np.column_stack([Z, conf, recon(Z)])
    Ftr = feats(Ztr, oof_conf)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced",
                                  random_state=seed).fit(Ftr, oof_err)
    gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=seed).fit(Ftr, oof_err)

    # Probability calibration is fitted on clean training data only, using CV.
    # "sigmoid" is Platt-style calibration; isotonic is a non-parametric alternative.
    calibrators = {}
    for method in ("sigmoid", "isotonic"):
        try:
            cal = CalibratedClassifierCV(estimator=clone(model), method=method, cv=3)
            cal.fit(Xtr, ytr)
            calibrators[method] = cal
        except Exception as exc:
            warnings.warn(f"{method} calibration failed for {name}/{model_name}/seed={seed}: {exc}")

    def evaluate(Xc, yc, kind, sev):
        proba = model.predict_proba(Xc); conf = proba.max(1); pred = proba.argmax(1)
        correct = (pred == yc); err = (~correct).astype(int); Z = pre.transform(Xc)
        F = feats(Z, conf)
        scores = {"confidence": 1 - conf, "pca_recon": recon(Z),
                  "tree": tree.predict_proba(F)[:, 1], "gbm": gbm.predict_proba(F)[:, 1]}
        ece_sigmoid = np.nan
        ece_isotonic = np.nan
        if "sigmoid" in calibrators:
            cp = calibrators["sigmoid"].predict_proba(Xc)
            ece_sigmoid = ece(cp.max(1), correct)
        if "isotonic" in calibrators:
            cp = calibrators["isotonic"].predict_proba(Xc)
            ece_isotonic = ece(cp.max(1), correct)
        common = dict(dataset=name, model=model_name, seed=seed, shift_type=kind, severity=sev,
                      accuracy=correct.mean(), macro_f1=f1_score(yc, pred, average="macro"),
                      error_rate=err.mean(), ece_raw=ece(conf, correct),
                      ece_sigmoid=ece_sigmoid, ece_isotonic=ece_isotonic)
        return [dict(common, predictor=p, **failure_metrics(s, err)) for p, s in scores.items()]

    rows = evaluate(Xte, yte, "clean", 0.0)
    stats = feature_stats(Xtr, num)
    for k, kind in enumerate(KINDS):
        for sev in SEVERITIES:
            rng = np.random.default_rng([seed, k, int(sev * 100)])
            rows += evaluate(corrupt(Xte, kind, sev, rng, num, cat, stats), yte, kind, sev)
    if Xsh is not None and len(Xsh) > 0:
        rows += evaluate(Xsh, ysh, "covariate_shift", 1.0)

    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["adult", "bank", "wine"])
    ap.add_argument("--models", nargs="+", default=["logreg", "svm", "rf", "gbm"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-rows", type=int, default=6000)
    ap.add_argument("--out", default="results/results.csv")
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()   # resume: skip finished work
    for name in a.datasets:
        X, y, shifted = load(name, a.max_rows)
        print(f"{name}: {len(X)} rows, positive rate {y.mean():.2f}, shifted group {shifted.mean():.0%}")
        for m in a.models:
            for seed in range(a.seeds):
                if len(done) and ((done.dataset == name) & (done.model == m) & (done.seed == seed)).any():
                    continue
                t = time.time(); rows = pd.DataFrame(run_one(X, y, shifted, m, seed, name))
                rows.to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"  {name}/{m}/seed{seed} done in {time.time()-t:.0f}s", flush=True)
