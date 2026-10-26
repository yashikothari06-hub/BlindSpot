"""Create a PCA 2D map showing correct vs incorrect predictions on held-out data.

Usage: python -m blindspot.pca_map --dataset adult --model rf --seed 0
The PCA visualization is diagnostic; the 2D projection can hide variance, so the
explained-variance percentages are included in the axes.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from .data import load
from .run import make_pre

def model_for(name, seed):
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "svm": SVC(probability=True, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "gbm": HistGradientBoostingClassifier(random_state=seed),
    }
    return models[name]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["adult", "bank", "wine", "synthetic"], default="adult")
    ap.add_argument("--model", choices=["logreg", "svm", "rf", "gbm"], default="rf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=6000)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    X, y, shifted = load(args.dataset, args.max_rows)
    in_idx = np.where(~shifted)[0]
    train_idx, clean_test_idx = train_test_split(
        in_idx, test_size=0.3, stratify=y[in_idx], random_state=args.seed
    )
    Xtr, ytr = X.iloc[train_idx], y[train_idx]
    Xclean, yclean = X.iloc[clean_test_idx], y[clean_test_idx]
    Xshift, yshift = X.iloc[np.where(shifted)[0]], y[np.where(shifted)[0]]

    num = list(X.select_dtypes("number").columns)
    cat = [c for c in X.columns if c not in num]
    prep = make_pre(num, cat)
    clf = Pipeline([("pre", prep), ("clf", model_for(args.model, args.seed))]).fit(Xtr, ytr)
    Ztr = clf.named_steps["pre"].transform(Xtr)
    # Separate 2-component PCA for visualization; fitted on clean training data only.
    n_comp = min(2, Ztr.shape[1], Ztr.shape[0] - 1)
    pca = PCA(n_components=n_comp, svd_solver="full").fit(Ztr)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    point_frames = []
    groups = [("clean_test", Xclean, yclean)]
    if len(Xshift):
        groups.append(("heldout_shift", Xshift, yshift))
    for condition, Xpart, ypart in groups:
        Z = clf.named_steps["pre"].transform(Xpart)
        coords = pca.transform(Z)
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
        pred = clf.predict(Xpart)
        frame = pd.DataFrame({
            "dataset": args.dataset, "model": args.model, "seed": args.seed,
            "condition": condition, "pc1": coords[:, 0], "pc2": coords[:, 1],
            "correct": (pred == ypart).astype(int),
            "confidence": clf.predict_proba(Xpart).max(axis=1),
        })
        point_frames.append(frame)
    points = pd.concat(point_frames, ignore_index=True)
    csv_path = out_dir / f"pca_map_{args.dataset}_{args.model}_seed{args.seed}.csv"
    points.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5), squeeze=False)
    ev = pca.explained_variance_ratio_
    for ax, (condition, _, _) in zip(axes[0], groups):
        part = points[points.condition == condition]
        for correct, label, marker in [(1, "Correct", "o"), (0, "Incorrect", "x")]:
            q = part[part.correct == correct]
            if len(q):
                ax.scatter(q.pc1, q.pc2, s=16, alpha=0.55, marker=marker, label=label)
        ax.set_title(f"{condition}: accuracy={part.correct.mean():.3f}")
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)" if len(ev) > 1 else "PC2 (0% variance; padded)")
        ax.legend()
    fig.suptitle(f"BlindSpot PCA failure map | {args.dataset} | {args.model}")
    fig.tight_layout()
    image_path = out_dir / f"pca_map_{args.dataset}_{args.model}_seed{args.seed}.png"
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    print(f"Saved PCA map: {image_path}")
    print(f"Saved PCA points: {csv_path}")
    print(f"Clean-test accuracy: {accuracy_score(yclean, clf.predict(Xclean)):.3f}")

if __name__ == "__main__":
    main()
