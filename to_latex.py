import pandas as pd

# In case of running it, you must pip install jinja2
# This is a side library used to convert the top 10 results into latex format
def generate_ranking(csv_path, dataset, columns_to_keep=None):
    df = pd.read_csv(csv_path)

    if "balanced_accuracy_mean" not in df.columns:
        raise ValueError("Column'balanced_accuracy_mean' not found")

    # Merge gfsir thresholds if both exist
    if "gfsir_minth" in df.columns and "gfsir_maxth" in df.columns:
        df["gfsir_th_range"] = df.apply(
            lambda row: f"[{row['gfsir_minth']}, {row['gfsir_maxth']}]"
            if pd.notna(row["gfsir_minth"]) and pd.notna(row["gfsir_maxth"])
            else None,
            axis=1
        )

    # If user specified columns to keep
    if columns_to_keep is not None:
        missing_cols = [c for c in columns_to_keep if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in the csv CSV: {missing_cols}")

        df = df[columns_to_keep]

    # Sorting by desc balanced accuracy
    df_sorted = df.sort_values(by="balanced_accuracy_mean", ascending=False)

    # Selects only the top
    top10 = df_sorted.head(10).copy()

    # Adding the rank column
    top10.insert(0, "rank", range(1, len(top10) + 1))

    # Generating as a rotated latex
    latex_table = top10.to_latex(
        index=False,
        float_format="%.4f",
        escape=True
    )

    print(r"\begin{sidewaystable}[ht]")
    print(r"\centering")
    print(r"\caption{Top 10 most performing selectors for "+dataset + r" sorted by balanced\_accuracy\_mean}")
    print(r"\label{tab:top10_balanced_accuracy}")
    print(latex_table)
    print(r"\end{sidewaystable}")

if __name__ == "__main__":

    columns_to_keep = [
        "selector",
        "similarity_function",
        "threshold",
        "cn_selector",
        "gfsir_nfeatures",
        # "gfsir_minth",
        # "gfsir_maxth",
        "gfsir_th_range",
        "gfsir_selector",
        "balanced_accuracy_mean",
        # "balanced_accuracy_std",
        "auroc_mean",
        # "auroc_std",
        "feature_stability",
        "runtime_mean",
        "features_mean"
    ]

    dataset = "radiomics_lgg"
    csv_path = f"outputs/{dataset}/{dataset}_benchmark_results.csv"
    generate_ranking(csv_path, dataset, columns_to_keep)
