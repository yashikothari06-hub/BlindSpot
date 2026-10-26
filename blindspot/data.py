"""Dataset loaders. Each returns (X: DataFrame, y: int array, shifted: bool array).
`shifted` marks the rows that form the genuine covariate-shift test group; they are
NEVER used for training. See README for how each shift is defined."""
import numpy as np, pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
              "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
              "hours-per-week", "native-country", "income"]

def _adult():
    df = pd.read_csv(RAW / "adult.data", header=None, names=ADULT_COLS,
                     na_values="?", skipinitialspace=True)
    y = (df.pop("income") == ">50K").astype(int).to_numpy()
    df = df.drop(columns=["fnlwgt"])                    # sampling weight, not a feature
    return df, y, (df["age"] >= 45).to_numpy()          # shift: train young, test older

def _bank():
    df = pd.read_csv(RAW / "bank-full.csv", sep=";")
    y = (df.pop("y") == "yes").astype(int).to_numpy()
    df = df.drop(columns=["duration"])                  # only known after the call: leakage
    # Define the time-related shift from the month feature, not file row order.
    # Earlier months are the in-distribution pool; Jul-Dec form the held-out group.
    # This is a month-conditioned shift stress test, not proof that P(Y|X) is invariant.
    month = df["month"].astype(str).str.lower()
    later_months = {"jul", "aug", "sep", "oct", "nov", "dec"}
    shifted = month.isin(later_months).to_numpy()
    return df, y, shifted

def _wine():
    red = pd.read_csv(RAW / "winequality-red.csv", sep=";"); red["wine_type"] = "red"
    white = pd.read_csv(RAW / "winequality-white.csv", sep=";"); white["wine_type"] = "white"
    df = pd.concat([red, white], ignore_index=True).drop_duplicates().reset_index(drop=True)
    y = (df.pop("quality") >= 7).astype(int).to_numpy() # fixed, pre-declared threshold
    shifted = (df.pop("wine_type") == "red").to_numpy() # shift: train white, test red
    return df, y, shifted

def _synthetic(n=4000, seed=0):
    """Offline smoke-test dataset: numeric + categorical features, shift on feature 0."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n, 8, n_informative=5, n_redundant=1, flip_y=0.08,
                               class_sep=0.8, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["c1"] = pd.cut(df["f1"], 3, labels=["a", "b", "c"]).astype(object)
    df["c2"] = pd.cut(df["f2"], 4, labels=["w", "x", "y", "z"]).astype(object)
    return df, y, (df["f0"] > df["f0"].quantile(0.7)).to_numpy()

LOADERS = {"adult": _adult, "bank": _bank, "wine": _wine, "synthetic": _synthetic}

def load(name, max_rows=6000):
    X, y, shifted = LOADERS[name]()
    for c in X.columns:                                 # categoricals -> object, NaN-friendly
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    if max_rows and len(X) > max_rows:                  # fixed subsample so all seeds share data
        keep = np.sort(np.random.default_rng(0).choice(len(X), max_rows, replace=False))
        X, y, shifted = X.iloc[keep].reset_index(drop=True), y[keep], shifted[keep]
    return X, y, shifted
