import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CN_SELECTORS = {"Label Propagation", "Bridging Centrality", "Louvain", "Structural Diversity"}
SIMILARITY_FUNCTIONS = ["Cosine", "Spearman", "Pearson", "Rho distance"]

def accuracy_vs_runtime_by_threshold(summary, dataset):
    df = summary.copy()

    # Group threshold by intervals
    bins = [-np.inf, 0.15, 0.30, 0.60, np.inf]
    thresh_labels = [
        "thresh ≤ 0.15",
        "0.15 < thresh ≤ 0.30",
        "0.30 < thresh ≤ 0.60",
        "thresh > 0.60"
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

    bins = [-np.inf, 0.15, 0.30, 0.60, np.inf]
    labels = [
        "thresh ≤ 0.15",
        "0.15 < thresh ≤ 0.30",
        "0.30 < thresh ≤ 0.60",
        "thresh > 0.60"
    ]

    # Split CN vs non-CN
    cn_df = df[df["selector"] == "DyGraFS"].copy()
    non_cn_df = df[df["selector"] != "DyGraFS"]

    cn_df["thresh_group"] = pd.cut(cn_df["threshold"], bins=bins, labels=labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, group in zip(axes, labels):

        for selector in non_cn_df["selector"].unique():
            sub_sel = non_cn_df[non_cn_df["selector"] == selector]

            ax.scatter(
                sub_sel["features_mean"],
                sub_sel["balanced_accuracy_mean"],
                c="steelblue",
                s=60,
                alpha=0.6,
                label=selector
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
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4)

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

        for selector in non_cn_df["selector"].unique():
            sub_sel = non_cn_df[non_cn_df["selector"] == selector]

            ax.scatter(
                sub_sel["features_mean"],
                sub_sel["balanced_accuracy_mean"],
                c="steelblue",
                s=60,
                alpha=0.6,
                label=selector
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
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4)

    fig.suptitle(f"{dataset}: Accuracy vs Features by CN Selector", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_features_by_cn_selector.png"
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