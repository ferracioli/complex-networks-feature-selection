import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# All figures plotted in english are defined here
CN_SELECTORS = {"Label Propagation", "Bridging Centrality", "Louvain", "Structural Diversity"}
SIMILARITY_FUNCTIONS = ["Cosine", "Spearman", "Pearson", "Rho distance"]

def accuracy_vs_runtime_by_threshold(summary, dataset):
    df = summary.copy()

    # Group threshold by intervals
    bins = [0.0, 0.25, 0.5, 0.75, 0.9]
    thresh_labels = [
        "0.0 ≤ thresh ≤ 0.25",
        "0.25 < thresh ≤ 0.5",
        "0.5 < thresh ≤ 0.75",
        "0.75 < thresh ≤ 0.9"
    ]

    df["thresh_group"] = pd.cut(df["threshold"], bins=bins, labels=thresh_labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, group in zip(axes, thresh_labels):

        non_cn = df[~df["cn_selector"].isin(CN_SELECTORS)]

        ax.scatter(
            non_cn["runtime_mean"],
            non_cn["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.6,
            label="Other selectors"
        )

        cn = df[
            (df["cn_selector"].isin(CN_SELECTORS)) &
            (df["thresh_group"] == group)
        ]

        ax.scatter(
            cn["runtime_mean"],
            cn["balanced_accuracy_mean"],
            c="orange",
            s=80,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label="DyGraFS"
        )

        ax.set_title(group)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Runtime (mean, seconds)")
    axes[3].set_xlabel("Runtime (mean, seconds)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"{dataset}: Accuracy vs Runtime by Threshold Range", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_runtime_by_threshold.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def accuracy_vs_runtime_by_similarity_function(summary, dataset):
    df = summary.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, similarity_function in zip(axes, SIMILARITY_FUNCTIONS):

        # Non-CN selectors
        non_cn = df[~df["cn_selector"].isin(CN_SELECTORS)]

        ax.scatter(
            non_cn["runtime_mean"],
            non_cn["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.6,
            label="Other selectors"
        )

        # CN selectors for this similarity_function only
        cn = df[
            (df["cn_selector"].isin(CN_SELECTORS)) &
            (df["similarity_function"] == similarity_function)
        ]

        ax.scatter(
            cn["runtime_mean"],
            cn["balanced_accuracy_mean"],
            c="orange",
            s=80,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label="DyGraFS"
        )

        ax.set_title(f"Similarity Function: {similarity_function}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Runtime (mean, seconds)")
    axes[3].set_xlabel("Runtime (mean, seconds)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"{dataset}: Accuracy vs Runtime by Similarity Function", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_runtime_by_similarity_function.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def accuracy_vs_runtime_by_cn_selector(summary, dataset):
    df = summary.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, cn_sel in zip(axes, sorted(CN_SELECTORS)):

        # Non-CN selectors
        non_cn = df[~df["cn_selector"].isin(CN_SELECTORS)]

        ax.scatter(
            non_cn["runtime_mean"],
            non_cn["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.6,
            label="Other selectors"
        )

        # Only this CN selector
        cn = df[df["cn_selector"] == cn_sel]

        ax.scatter(
            cn["runtime_mean"],
            cn["balanced_accuracy_mean"],
            c="orange",
            s=80,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label=f"DyGraFS"
        )
        ax.set_title(f"CN selector: {cn_sel}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Runtime (mean, seconds)")
    axes[3].set_xlabel("Runtime (mean, seconds)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"{dataset}: Accuracy vs Runtime by CN Selector", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_runtime_by_cn_selector.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def performance_boxplot(summary, dataset, metric="balanced_accuracy"):
    """
    Renders mean performance with 95% Confidence Interval error bars per feature selector.
    """
    df = summary.copy().sort_values(f"{metric}_mean", ascending=False)

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(df))
    
    plt.errorbar(
        x_positions,
        df[f"{metric}_mean"],
        yerr=df[f"{metric}_ci95"],
        fmt='o',
        color='darkblue',
        ecolor='crimson',
        elinewidth=2,
        capsize=5,
        markersize=8,
        label='Mean ± 95% CI'
    )

    plt.xticks(x_positions, df["selector"], rotation=30, ha="right")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{dataset}: {metric.replace('_', ' ').title()} with 95% Confidence Intervals")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = f"outputs/{dataset}/{dataset}_boxplot_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def feature_stability_plot(summary, dataset):
    """
    Plots Feature Stability (Jaccard Index across CV folds) per selector.
    """
    if "feature_stability" not in summary.columns:
        print("Warning: 'feature_stability' column not found in summary. Skipping plot.")
        return

    # Drop entries where feature_stability is NaN
    df = summary.dropna(subset=["feature_stability"]).copy()
    if df.empty:
        return

    df = df.sort_values("feature_stability", ascending=False)

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(df))

    plt.bar(
        x_positions,
        df["feature_stability"],
        color="teal",
        alpha=0.8,
        edgecolor="black"
    )

    plt.xticks(x_positions, df["selector"], rotation=30, ha="right")
    plt.ylabel("Feature Stability (Mean Jaccard Index)")
    plt.title(f"{dataset}: Feature Stability across CV Folds")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = f"outputs/{dataset}/{dataset}_feature_stability.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def accuracy_vs_features_by_similarity_function(summary, dataset):
    df = summary.copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    cn_df = df[df["selector"] == "DyGraFS"]
    non_cn_df = df[df["selector"] != "DyGraFS"]

    for ax, similarity_function in zip(axes, SIMILARITY_FUNCTIONS):
        # Plot all other selectors under one unified label
        ax.scatter(
            non_cn_df["features_mean"],
            non_cn_df["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.6,
            label="Other selectors"
        )

        cn_similarity_function = cn_df[cn_df["similarity_function"] == similarity_function]
        ax.scatter(
            cn_similarity_function["features_mean"],
            cn_similarity_function["balanced_accuracy_mean"],
            c="orange",
            s=80,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label="DyGraFS"
        )
        ax.set_title(f"Similarity Function: {similarity_function}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Mean Number of Selected Features")
    axes[3].set_xlabel("Mean Number of Selected Features")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=2)

    fig.suptitle(f"{dataset}: Accuracy vs Features by Similarity Function", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"outputs/{dataset}/{dataset}_accuracy_vs_features_by_similarity_function.png", dpi=300)
    plt.close()

def accuracy_vs_features_by_threshold(summary, dataset):
    df = summary.copy()

    bins = [0.0, 0.25, 0.5, 0.75, 0.9]
    labels = [
        "0.0 ≤ thresh ≤ 0.25",
        "0.25 < thresh ≤ 0.5",
        "0.5 < thresh ≤ 0.75",
        "0.75 < thresh ≤ 0.9"
    ]

    # Split CN vs non-CN
    cn_df = df[df["selector"] == "DyGraFS"].copy()
    non_cn_df = df[df["selector"] != "DyGraFS"]

    cn_df["thresh_group"] = pd.cut(cn_df["threshold"], bins=bins, labels=labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, group in zip(axes, labels):

        ax.scatter(
            non_cn_df["features_mean"],
            non_cn_df["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.6,
            label="Other selectors"
        )

        sub_cn = cn_df[cn_df["thresh_group"] == group]

        ax.scatter(
            sub_cn["features_mean"],
            sub_cn["balanced_accuracy_mean"],
            c="orange",
            s=80,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label="DyGraFS"
        )

        ax.set_title(group)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Mean Number of Selected Features")
    axes[3].set_xlabel("Mean Number of Selected Features")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=2)

    fig.suptitle(f"{dataset}: Accuracy vs Features by Threshold Range", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_features_by_threshold.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def accuracy_vs_features_by_cn_selector(summary, dataset):
    """
    Balanced Accuracy vs mean number of selected features,
    separated by CN selector (one subplot per CN selector)
    """
    df = summary.copy()

    cn_df = df[df["selector"] == "DyGraFS"]
    non_cn_df = df[df["selector"] != "DyGraFS"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, cn_sel in zip(axes, sorted(CN_SELECTORS)):

        ax.scatter(
            non_cn_df["features_mean"],
            non_cn_df["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.6,
            label="Other selectors"
        )

        sub_cn = cn_df[cn_df["cn_selector"] == cn_sel]

        ax.scatter(
            sub_cn["features_mean"],
            sub_cn["balanced_accuracy_mean"],
            c="orange",
            s=80,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label="DyGraFS"
        )

        ax.set_title(f"CN selector: {cn_sel}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Mean Number of Selected Features")
    axes[3].set_xlabel("Mean Number of Selected Features")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=2)

    fig.suptitle(f"{dataset}: Accuracy vs Features by CN Selector", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_features_by_cn_selector.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def dygrafs_param_heatmap(dygrafs_inner_summary, dataset, metric="balanced_accuracy_mean"):
    """
    Heatmap of DyGraFS's own inner-CV grid: similarity_function x cn_selector,
    averaged over `threshold` (and over outer folds, already done upstream).

    Unlike the "DyGraFS vs everyone else" scatter plots, this isolates DyGraFS's
    internal hyperparameter sensitivity in one glance, which the scatter plots
    (faceted one dimension at a time) don't show directly: e.g. it makes it
    immediately visible that 'Bridging Centrality' underperforms every other
    cn_selector across every similarity_function.
    """
    df = dygrafs_inner_summary.copy()
    if df.empty or "cn_selector" not in df.columns or "similarity_function" not in df.columns:
        print("Warning: dygrafs_inner_summary missing expected columns. Skipping heatmap.")
        return

    pivot = df.pivot_table(
        index="cn_selector", columns="similarity_function", values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, label=metric.replace("_", " ").title())
    ax.set_title(f"{dataset}: DyGraFS Inner-CV {metric.replace('_', ' ').title()}\nby CN Selector x Similarity Function")
    fig.tight_layout()

    out_path = f"outputs/{dataset}/{dataset}_dygrafs_param_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def print_cn_performance_summary(outfile, summary):
    """
    Print aggregated performance statistics for Complex Network selectors
    in a single table, including an 'all' row for overall performance.
    Also saves the output to a txt file.
    """

    df = summary.copy()

    # Keep only DyGraFS(Complex Network) runs
    df = df[df["selector"] == "DyGraFS"]

    if df.empty:
        text = "No DyGraFS selectors found in summary."
        print(text)
        with open(outfile, "a") as f:
            f.write(text + "\n")
        return

    metric_col = f"balanced_accuracy_mean"

    # Create a copy with a fake group called "all" for overall stats
    df_all = df.copy()
    df_all["cn_selector"] = "all"

    # Combine original + overall
    df_combined = pd.concat([df, df_all], ignore_index=True)

    # Group and aggregate
    stats = (
        df_combined
        .groupby("cn_selector")[metric_col]
        .agg([
            ("mean", "mean"),
            ("std", "std"),
            ("median", "median"),
            ("min", "min"),
            ("max", "max"),
            ("n_runs", "count")
        ])
        .sort_values("cn_selector", ascending=False)
    )

    header = "\n===== Complex Network Performance Summary ====="
    table = stats.round(4).to_string()
    footer = "==============================================\n"

    # Print to console
    print(header)
    print(table)
    print(footer)

    # Save to txt file
    with open(outfile, "a") as f:
        f.write(header + "\n")
        f.write(table + "\n")
        f.write(footer)

# pipeline/model_plots.py

def accuracy_vs_threshold_by_cn_selector(summary, dataset):
    """
    Generates 4 subplots (one for each CN selector) showing 
    Balanced Accuracy vs. Threshold, replicating 'Other selectors' in every subplot.
    """
    df = summary.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    # Pre-extract non-DyGraFS selectors once
    non_cn = df[~df["cn_selector"].isin(CN_SELECTORS)]

    for ax, cn_sel in zip(axes, sorted(CN_SELECTORS)):
        # 1. Plot "Other selectors" identically in all subplots
        ax.scatter(
            non_cn["threshold"],
            non_cn["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.4,
            label="Other selectors"
        )

        # 2. Plot specific DyGraFS CN selector for this subplot
        cn = df[df["cn_selector"] == cn_sel]
        ax.scatter(
            cn["threshold"],
            cn["balanced_accuracy_mean"],
            c="orange",
            s=80,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label="DyGraFS"
        )

        ax.set_title(f"CN selector: {cn_sel}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Threshold")
    axes[3].set_xlabel("Threshold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"{dataset}: Accuracy vs. Threshold by CN Selector", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_threshold_by_cn_selector.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # ============================================================
# Reporting helpers: Overleaf table + feature selection frequency
# ============================================================
def _latex_escape(value):
    """Escape characters that have a special meaning in LaTeX."""
    if pd.isna(value):
        return "--"

    value = str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return value


def save_overleaf_benchmark_table(
    df,
    output_path,
    ranking_metric="balanced_accuracy_mean"
):
    """
    Generate a pre-filled LaTeX table directly from the benchmark CSV.

    Rows are ranked globally using the requested metric, so the same
    ranking criterion is used consistently across selectors.
    """

    # Higher balanced accuracy = better rank.
    df = df.sort_values(
        by=ranking_metric,
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    # Keep the table compact and publication-oriented.
    preferred_columns = [
        "Rank",
        "selector",
        "balanced_accuracy_mean",
        "balanced_accuracy_ci95",
        "features_mean",
        "feature_stability",
        "runtime_mean",
        "p_value_vs_dygrafs_holm",
    ]

    columns = [c for c in preferred_columns if c in df.columns]
    table_df = df[columns].copy()

    # Human-readable Overleaf column names.
    column_names = {
        "Rank": "Rank",
        "selector": "Selector",
        "balanced_accuracy_mean": "Bal. Acc.",
        "balanced_accuracy_ci95": "95\\% CI",
        "features_mean": "Features",
        "feature_stability": "Stability",
        "runtime_mean": "Runtime (s)",
        "p_value_vs_dygrafs_holm": "p-value (Holm)",
    }

    # Formatting keeps the generated table ready to paste into Overleaf.
    for col in table_df.columns:
        if col in {
            "balanced_accuracy_mean",
            "balanced_accuracy_ci95",
            "feature_stability",
            "p_value_vs_dygrafs_holm",
        }:
            table_df[col] = table_df[col].map(
                lambda x: "--" if pd.isna(x) else f"{float(x):.4f}"
            )

        elif col == "features_mean":
            table_df[col] = table_df[col].map(
                lambda x: "--" if pd.isna(x) else f"{float(x):.1f}"
            )

        elif col == "runtime_mean":
            table_df[col] = table_df[col].map(
                lambda x: "--" if pd.isna(x) else f"{float(x):.1f}"
            )

        elif col == "Rank":
            table_df[col] = table_df[col].astype(int).astype(str)

        else:
            table_df[col] = table_df[col].map(_latex_escape)

    table_df = table_df.rename(columns=column_names)

    # Build the actual LaTeX tabular environment.
    latex_lines = []

    latex_lines.append(
        f"% Ranking metric: {ranking_metric} (descending)"
    )
    latex_lines.append("% Paste directly into an Overleaf document.")
    latex_lines.append("")
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Benchmark comparison of feature selectors.}")
    latex_lines.append(r"\label{tab:feature_selector_benchmark}")

    # All columns centered; adjust if desired in Overleaf.
    latex_lines.append(
        r"\begin{tabular}{"
        + "c" * len(table_df.columns)
        + "}"
    )
    latex_lines.append(r"\toprule")

    header = " & ".join(
        [str(c) for c in table_df.columns]
    ) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")

    for _, row in table_df.iterrows():
        values = [
            _latex_escape(v)
            for v in row.tolist()
        ]
        latex_lines.append(" & ".join(values) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"Saved Overleaf-ready table to: {output_path}")

def save_feature_selection_frequency(results, output_dir, dataset):
    """
    Save feature-selection frequency for every selector.

    Produces:
      1. One CSV per selector.
      2. One combined CSV for all selectors.
      3. A compact TXT summary.

    Frequency is based on the outer CV folds.
    """

    all_frequency_rows = []

    for result in results:
        selector = result.get("selector", "Unknown")
        frequency = result.get("feature_selection_frequency", {})
        frequency_pct = result.get("feature_selection_frequency_pct", {})

        if not frequency:
            continue

        rows = []

        for feature, count in frequency.items():
            pct = frequency_pct.get(feature, np.nan)

            rows.append({
                "Selector": selector,
                "Feature": feature,
                "Selection_Count": count,
                "Selection_Frequency_Pct": pct,
            })

            all_frequency_rows.append({
                "Selector": selector,
                "Feature": feature,
                "Selection_Count": count,
                "Selection_Frequency_Pct": pct,
            })

        frequency_df = pd.DataFrame(rows)

        frequency_df = frequency_df.sort_values(
            by=["Selection_Count", "Feature"],
            ascending=[False, True]
        ).reset_index(drop=True)

        # Add an explicit rank.
        frequency_df.insert(
            0,
            "Frequency_Rank",
            np.arange(1, len(frequency_df) + 1)
        )

        safe_selector_name = (
            selector.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )

        frequency_df.to_csv(f"{output_dir}/{dataset}_selector_{safe_selector_name}_feature_frequency.csv", index=False)

    # Combined table across all selectors.
    if all_frequency_rows:
        combined_df = pd.DataFrame(all_frequency_rows)

        combined_df = combined_df.sort_values(
            by=["Feature", "Selection_Frequency_Pct"],
            ascending=[True, False]
        ).reset_index(drop=True)

        combined_df.to_csv(f"{output_dir}/{dataset}_feature_selection_frequency_all.csv", index=False)

        # Pivoted form is especially convenient for comparing selectors.
        frequency_matrix = combined_df.pivot_table(
            index="Feature",
            columns="Selector",
            values="Selection_Frequency_Pct",
            fill_value=0
        )

        frequency_matrix = frequency_matrix.sort_index()

        frequency_matrix.to_csv(f"{output_dir}/{dataset}_feature_selection_frequency_matrix.csv")

        # Compact human-readable report.
        txt_path = (
            f"{output_dir}/{dataset}_feature_selection_frequency_summary.txt"
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Feature-selection frequency summary: {dataset}\n")
            f.write("=" * 80 + "\n\n")

            for selector in frequency_matrix.columns:
                f.write(f"\n[{selector}]\n")
                f.write("-" * 80 + "\n")

                selector_df = combined_df[
                    combined_df["Selector"] == selector
                ].sort_values(
                    by="Selection_Frequency_Pct",
                    ascending=False
                )

                for _, row in selector_df.iterrows():
                    f.write(
                        f"{row['Feature']:<60} "
                        f"{row['Selection_Count']:>2} / "
                        f"{len(results[0].get('outer_bal_accs_folds', [])):>2} "
                        f"({row['Selection_Frequency_Pct']:>6.1f}%)\n"
                    )

        print(f"Saved feature-selection frequency reports to: {output_dir}")
        