import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CN_SELECTORS = {"Label Propagation", "Bridging Centrality", "Louvain", "Structural Diversity"}
LINK_METHODS = ["Cosine", "Spearman", "Pearson", "Rho distance"]

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
            label="Non-CN"
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
            label="CN"
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

def accuracy_vs_runtime_by_link_method(summary, dataset):
    df = summary.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, link in zip(axes, LINK_METHODS):

        # Non-CN selectors
        non_cn = df[~df["cn_selector"].isin(CN_SELECTORS)]

        ax.scatter(
            non_cn["runtime_mean"],
            non_cn["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.6,
            label="Non-CN"
        )

        # CN selectors for this link_method only
        cn = df[
            (df["cn_selector"].isin(CN_SELECTORS)) &
            (df["link_method"] == link)
        ]

        ax.scatter(
            cn["runtime_mean"],
            cn["balanced_accuracy_mean"],
            c="orange",
            s=80,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label="CN"
        )

        ax.set_title(f"Link method: {link}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Runtime (mean, seconds)")
    axes[3].set_xlabel("Runtime (mean, seconds)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"{dataset}: Accuracy vs Runtime by Link Method", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_runtime_by_link_method.png"
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
            label="Non-CN"
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
            label=f"CN"
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

def accuracy_vs_features_by_link_method(summary, dataset):
    df = summary.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    cn_df = df[df["selector"] == "Complex Network"]
    non_cn_df = df[df["selector"] != "Complex Network"]

    for ax, link in zip(axes, LINK_METHODS):

        # Non-CN selectors (independent of link method)
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

        # CN selectors for this link method
        cn_link = cn_df[cn_df["link_method"] == link]

        ax.scatter(
            cn_link["features_mean"],
            cn_link["balanced_accuracy_mean"],
            c="orange",
            s=80,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label="Complex Network"
        )

        ax.set_title(f"Link method: {link}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_ylabel("Balanced Accuracy (mean)")
    axes[2].set_xlabel("Mean Number of Selected Features")
    axes[3].set_xlabel("Mean Number of Selected Features")

    # De-duplicate legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4)

    fig.suptitle(f"{dataset}: Accuracy vs Features by Link Method", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_features_by_link_method.png"
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
    cn_df = df[df["selector"] == "Complex Network"].copy()
    non_cn_df = df[df["selector"] != "Complex Network"]

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
            label="Complex Network"
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

    cn_df = df[df["selector"] == "Complex Network"]
    non_cn_df = df[df["selector"] != "Complex Network"]

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
            label="Complex Network"
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

def print_cn_performance_summary(summary, metric="balanced_accuracy"):
    """
    Print aggregated performance statistics for Complex Network selectors
    in a single table, including an 'all' row for overall performance.
    """

    df = summary.copy()

    # Keep only Complex Network runs
    df = df[df["selector"] == "Complex Network"]

    if df.empty:
        print("No complex network selectors found in summary.")
        return

    metric_col = f"{metric}_mean"

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

    print("\n===== Complex Network Performance Summary =====")
    print(stats.round(4))
    print("==============================================\n")
