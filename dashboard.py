from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="BlindSpot | Failure Analysis", layout="wide")
st.title("BlindSpot")
st.subheader("Predicting and explaining machine-learning failure under distribution shift")
st.caption("Research dashboard: inspect results only after running the experiments and analysis.")

st.markdown("---")
st.header("Custom Dataset Experiment")
st.write("Upload a labelled CSV to evaluate a classifier under controlled data corruption.")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    key="custom_dataset_upload"
)

if uploaded_file is not None:
    custom_df = pd.read_csv(uploaded_file)

    st.write("Dataset preview")
    st.dataframe(custom_df.head(10), use_container_width=True)

    target_col = st.selectbox(
        "Select the target column",
        options=custom_df.columns.tolist(),
        key="custom_target"
    )

    model_name = st.selectbox(
        "Choose a classifier",
        options=["logreg", "svm", "rf", "gbm"],
        format_func=lambda x: {
            "logreg": "Logistic Regression",
            "svm": "Support Vector Machine",
            "rf": "Random Forest",
            "gbm": "Gradient Boosting"
        }[x],
        key="custom_model"
    )

    st.success("CSV uploaded and read successfully!")
    st.write("Number of rows:", len(custom_df))
    st.write("Number of columns:", len(custom_df.columns))
    st.write("Column names:", custom_df.columns.tolist())
    
    st.markdown("### Run your experiment")

    if st.button("Run Custom Experiment", type="primary"):
        from blindspot.run import run_one

        try:
            # Separate the target from the input features.
            X_custom = custom_df.drop(columns=[target_col]).copy()
            y_custom = custom_df[target_col].copy()

            # Remove rows with missing target labels.
            valid = y_custom.notna()
            X_custom = X_custom.loc[valid].reset_index(drop=True)
            y_custom = y_custom.loc[valid].reset_index(drop=True)

            # Convert target labels to integer class IDs.
            st.write("Selected target column:", target_col)
            st.write("Original target counts:", y_custom.value_counts(dropna=False).to_dict())

            y_custom, class_labels = pd.factorize(y_custom)

            st.write("Encoded class labels:", list(class_labels))
            st.write("Rows per class:", pd.Series(y_custom).value_counts().to_dict())

            if len(class_labels) < 2:
                st.error("The target column must contain at least two classes.")
            elif pd.Series(y_custom).value_counts().min() < 5:
                st.error("Each target class needs at least 5 rows.")
            else:
                with st.spinner("Training model and evaluating data shifts..."):
                    custom_results = run_one(
                        X_custom,
                        y_custom,
                        None,
                        model_name,
                        seed=42,
                        name="custom_upload"
                    )

                custom_results_df = pd.DataFrame(custom_results)
                st.write("Current working directory:", Path.cwd())
                custom_results_path = Path("results/custom_results.csv")
                custom_results_path.parent.mkdir(parents=True, exist_ok=True)
                custom_results_df.to_csv(custom_results_path, index=False)

                st.success(f"Results saved to {custom_results_path}")


                st.success("Custom experiment completed!")
                st.subheader("Custom experiment results")
                st.dataframe(
                    custom_results_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download experiment results",
                    data=custom_results_df.to_csv(index=False),
                    file_name="blindspot_custom_results.csv",
                    mime="text/csv"
                )

        except Exception as exc:
            st.error(f"Experiment failed: {exc}")


results_path = Path("results/results.csv")
summary_path = Path("results/summary.csv")
stats_path = Path("results/stats_vs_confidence.csv")

if not results_path.exists():
    st.warning("No experiment results found. Run `python -m blindspot.run --datasets synthetic --seeds 1` first, then run the analysis command.")
    st.stop()

df = pd.read_csv(results_path)
if df.empty:
    st.warning("The results file is empty.")
    st.stop()



ds_opts = sorted(df["dataset"].dropna().unique().tolist())
model_opts = sorted(df["model"].dropna().unique().tolist())
pred_opts = sorted(df["predictor"].dropna().unique().tolist())
shift_opts = sorted(df["shift_type"].dropna().unique().tolist())

def reset_filters():
    st.session_state["filter_datasets"] = ds_opts.copy()
    st.session_state["filter_models"] = model_opts.copy()
    st.session_state["filter_predictors"] = pred_opts.copy()
    st.session_state["filter_shifts"] = shift_opts.copy()

with st.sidebar:
    st.header("Experiment Filters")

    st.button(
        "Reset filters",
        on_click=reset_filters,
        use_container_width=True
    )

    datasets = st.multiselect(
        "Datasets", ds_opts,
        default=ds_opts,
        key="filter_datasets"
    )

    models = st.multiselect(
        "Base classifiers", model_opts,
        default=model_opts,
        key="filter_models"
    )

    predictors = st.multiselect(
        "Failure scores", pred_opts,
        default=pred_opts,
        key="filter_predictors"
    )

    shifts = st.multiselect(
        "Test conditions", shift_opts,
        default=shift_opts,
        key="filter_shifts"
    )

    st.caption(f"{len(datasets)} datasets · {len(models)} models")



f = df[df.dataset.isin(datasets) & df.model.isin(models) &
       df.predictor.isin(predictors) & df.shift_type.isin(shifts)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Experiment rows", f"{len(f):,}")
c2.metric("Datasets", str(f["dataset"].nunique()))
c3.metric("Base classifiers", str(f["model"].nunique()))
c4.metric("Seeds", str(f["seed"].nunique()))

st.markdown("### Classification performance")
if f.empty:
    st.info("No rows match these filters.")
else:
    metric = st.selectbox("Metric", ["accuracy", "macro_f1", "error_rate", "ece_raw", "ece_sigmoid", "ece_isotonic"])
    grouped = f.groupby(["dataset", "model", "shift_type", "severity"], as_index=False)[metric].mean()
    st.dataframe(grouped, use_container_width=True, hide_index=True)
    plot_df = grouped.groupby(["dataset", "model", "shift_type"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    for (dataset, model), part in plot_df.groupby(["dataset", "model"]):
        part = part.set_index("shift_type").reindex(shift_opts).dropna(subset=[metric]).reset_index()
        ax.plot(part["shift_type"], part[metric], marker="o", label=f"{dataset} / {model}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=40)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

st.markdown("### Failure-prediction performance")
fmetric = st.selectbox("Failure metric", ["auroc", "auprc", "recall_p50", "recall_top20", "risk80", "aurc"])
if not f.empty:
    fail_group = f.groupby(["dataset", "model", "shift_type", "predictor"], as_index=False)[fmetric].mean()
    st.dataframe(fail_group, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for (dataset, model, predictor), part in fail_group.groupby(["dataset", "model", "predictor"]):
        part = part.set_index("shift_type").reindex(shift_opts).dropna(subset=[fmetric]).reset_index()
        ax.plot(part["shift_type"], part[fmetric], marker="o", label=f"{dataset} / {model} / {predictor}")
    ax.set_ylabel(fmetric)
    ax.tick_params(axis="x", rotation=40)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

st.markdown("### AUPRC reference baseline")
st.write("For failure prediction, the positive class is a base-model error. Random-ranking AUPRC is approximately the error rate. The CSV's `error_rate` column is the reference baseline; compare AUPRC to that value or use the lift reported by the analysis script.")

if summary_path.exists():
    st.markdown("### Summary with confidence intervals")
    summ = pd.read_csv(summary_path)
    st.dataframe(summ, use_container_width=True, hide_index=True)

if stats_path.exists():
    st.markdown("### Paired comparison against confidence thresholding")
    stats = pd.read_csv(stats_path)
    st.dataframe(stats, use_container_width=True, hide_index=True)

st.caption("Interpret group-held-out tests as shift stress tests, not proof of a formal covariate-shift assumption. Results are experimental and should not be presented as established findings until the full run is complete.")


st.markdown("### PCA failure maps")
st.write("Generate a PCA error map with `python -m blindspot.pca_map --dataset adult --model rf --seed 0`. The map reports explained variance on its axes; a 2D projection can hide structure in omitted components.")
pca_images = sorted(Path("results").glob("pca_map_*.png"))
if pca_images:
    image_names = [p.name for p in pca_images]
    selected_image = st.selectbox("PCA map", image_names)
    st.image(str(Path("results") / selected_image), caption=selected_image, use_container_width=True)
else:
    st.info("No PCA maps generated yet.")
