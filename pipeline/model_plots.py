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
    Figure 1: Boxplot of performance metric by selector
    """
    df = summary.copy()

    # Boxplot of folds mean
    plt.figure(figsize=(10, 6))

    order = (
        df.sort_values(f"{metric}_mean", ascending=False)["selector"]
        .unique()
    )

    plt.boxplot(
        [df[df["selector"] == sel][f"{metric}_mean"] for sel in order],
        labels=order,
        showfliers=True
    )

    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{dataset}: {metric.replace('_', ' ').title()} Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = f"outputs/{dataset}/{dataset}_boxplot_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def accuracy_vs_features_by_similarity_function(summary, dataset):
    df = summary.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    cn_df = df[df["selector"] == "DyGraFS"]
    non_cn_df = df[df["selector"] != "DyGraFS"]

    for ax, similarity_function in zip(axes, SIMILARITY_FUNCTIONS):

        # Non-CN selectors (independent of Similarity Function)
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

        # CN selectors for this Similarity Function
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

    # De-duplicate legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4)

    fig.suptitle(f"{dataset}: Accuracy vs Features by Similarity Function", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_features_by_similarity_function.png"
    plt.savefig(out_path, dpi=300)
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

def accuracy_vs_threshold_by_cn_selector(summary, dataset):
    """
    Generates 4 subplots (one for each CN selector) showing 
    Balanced Accuracy vs. Threshold.
    """
    df = summary.copy()

    # Create the 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    # Iterate through selectors (assuming CN_SELECTORS is a predefined list of 4)
    for ax, cn_sel in zip(axes, sorted(CN_SELECTORS)):

        # 1. Plot "Other selectors" as background reference (Steelblue)
        non_cn = df[~df["cn_selector"].isin(CN_SELECTORS)]
        ax.scatter(
            non_cn["threshold"],
            non_cn["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.4, # Slightly more transparent to emphasize the target
            label="Other selectors"
        )

        # 2. Plot the specific CN selector for this subplot (Orange)
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

    # Add axis labels to the outer plots
    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Threshold")
    axes[3].set_xlabel("Threshold")

    # Handle the Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    # Main Title and Layout
    fig.suptitle(f"{dataset}: Accuracy vs. Threshold by CN Selector", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the output
    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_threshold_by_cn_selector.png"
    plt.savefig(out_path, dpi=300)
    plt.close()