# BlindSpot: Predicting and Explaining Machine Learning Failure Under Distribution Shift

**Research question.** Can we identify which inputs a classifier is likely to get wrong, and do
models with similar clean accuracy behave differently when the input distribution changes?

# BlindSpot

**Live Demo:** [Open BlindSpot Dashboard](https://blindspot-yashi.streamlit.app/)

BlindSpot is a machine learning research dashboard for exploring model performance, calibration, prediction failures, and data shift.


## Quick start
```bash
pip install -r requirements.txt
python download_data.py                      # UCI data -> data/raw/ (or place files manually)
python -m blindspot.run --seeds 1 --max-rows 3000   # first run: check it works and time it
python -m blindspot.run --seeds 10                   # full run (resumes if interrupted)
python -m blindspot.analyze results/results.csv      # tables + figures
streamlit run dashboard.py                           # interactive results dashboard
python -m blindspot.pca_map --dataset adult --model rf --seed 0  # PCA error map
python -m blindspot.run --datasets synthetic --seeds 1   # offline smoke test, no downloads
```
Expected files in `data/raw/`: `adult.data`, `bank-full.csv`, `winequality-red.csv`, `winequality-white.csv`.

## Design
| Step | What happens | Why |
|---|---|---|
| Split | Stratified 70/30 on the in-distribution group; the held-out group is never trained on | controlled group/domain shift evaluation |
| Preprocessing | Imputer, scaler, one-hot fit inside the pipeline on clean training data only | no leakage |
| OOF | 5-fold out-of-fold probabilities on the training set | honest error labels; an in-sample forest barely errs |
| PCA | Fit on clean training features (90% variance); reconstruction distance | explained variance, eigenvectors, anomaly score |
| Failure predictors | confidence (1 - max prob), PCA reconstruction error, shallow tree, gradient boosting | tree/GBM use features + OOF confidence + recon distance |
| Calibration | Platt-style sigmoid and isotonic probability calibration; ECE before/after | fit using clean training CV only |
| Conditions | clean, noise, missing, outliers (severities 0.2/0.5/1.0), covariate shift | corruption and shift kept separate |

**Corruptions (test data only).** noise: N(0, (s·std)²) on numeric cells; missing: 0.4·s of all cells blanked;
outliers: 0.1·s of numeric cells set to mean ± 6 std.

**Shifts.** Adult: train age < 45, test age ≥ 45. Bank: train on Jan-Jun and evaluate the held-out Jul-Dec month group (not based on file row order). Wine: train on white wine and evaluate on red wine. These are group-conditioned shift/domain stress tests; they do not prove a formal covariate-shift assumption.
Bank drops `duration` (known only after the call); Wine is de-duplicated, target = quality ≥ 7.

**Metrics.** accuracy, macro-F1, ECE (raw/calibrated); failure prediction: AUROC, AUPRC (also reported
as lift over the error-rate baseline), recall at precision 0.5, recall when flagging the top 20%,
selective risk at 80% coverage, area under the risk-coverage curve.
**Statistics.** Mean over seeds, bootstrap 95% CI, paired Wilcoxon of each predictor vs the confidence
baseline, Holm-corrected. Needs at least 5 seeds; use 10.

## Known limitations
- Seeds are repeated splits of the same finite data, so they are not independent samples.
- The failure predictors see PCA distance computed from a PCA fit on the same training rows (slightly optimistic in-sample).
- Datasets are subsampled to `--max-rows` (default 6000) to keep SVM tractable.
- Risk-coverage is summarized (risk@80%, AURC); full curves are not plotted yet.
- The dashboard reads `results/results.csv` and the analysis outputs; run the experiment and analysis steps first.

## Layout
`blindspot/data.py` loaders and shift definitions · `shifts.py` corruptions · `metrics.py` · `run.py` experiments · `analyze.py` stats and figures · `dashboard.py` interactive results view · `tests/`
