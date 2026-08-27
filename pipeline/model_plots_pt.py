import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# TODO
# performance_boxplot (traduzir o novo)
# feature_stability_plot

CN_SELECTORS = {"Label Propagation", "Bridging Centrality", "Louvain", "Structural Diversity"}
SIMILARITY_FUNCTIONS = ["Cosine", "Spearman", "Pearson", "Rho distance"]

def accuracy_vs_runtime_by_threshold_pt(summary, dataset):
    df = summary.copy()

    # Group threshold por intervals
    bins = [-np.inf, 0.15, 0.30, 0.60, np.inf]
    thresh_labels = [
        "limiar ≤ 0.15",
        "0.15 < limiar ≤ 0.30",
        "0.30 < limiar ≤ 0.60",
        "limiar > 0.60"
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
            label="Outros seletores"
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

    axes[0].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_xlabel("Tempo de execução (média, segundos)")
    axes[3].set_xlabel("Tempo de execução (média, segundos)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"{dataset.replace('four_class_nsclc', 'nsclc_quatro_classes')}: Acurácia balanceada vs Tempo por Intervalo de limiar", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_runtime_by_threshold_pt.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def accuracy_vs_runtime_by_similarity_function_pt(summary, dataset):
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
            label="Outros seletores"
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

        ax.set_title(f"Função de similaridade: {similarity_function}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_xlabel("Tempo de execução (média, segundos)")
    axes[3].set_xlabel("Tempo de execução (média, segundos)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"{dataset.replace('four_class_nsclc', 'nsclc_quatro_classes')}: Acurácia balanceada vs Tempo por Função de similaridade", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_runtime_by_similarity_function_pt.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def accuracy_vs_runtime_by_cn_selector_pt(summary, dataset):
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
            label="Outros seletores"
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
        ax.set_title(f"Seletor de rede: {cn_sel}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_xlabel("Tempo de execução (média, segundos)")
    axes[3].set_xlabel("Tempo de execução (média, segundos)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"{dataset.replace('four_class_nsclc', 'nsclc_quatro_classes')}: Acurácia balanceada vs Tempo por Seletor de rede", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_runtime_by_cn_selector_pt.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def performance_boxplot_pt(summary, dataset, metric="balanced_accuracy"):
    """
    Figure 1: Boxplot of performance metric por selector
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

    plt.ylabel(metric.replace('balanced_accuracy', 'Acurácia balanceada').replace("_", " ").title())
    plt.title(f"{dataset.replace('four_class_nsclc', 'nsclc_quatro_clases')}: Distribuição de {metric.replace('balanced_accuracy', 'Acurácia balanceada').replace('_', ' ').title()}")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = f"outputs/{dataset}/{dataset}_boxplot_{metric}_pt.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def accuracy_vs_features_by_similarity_function_pt(summary, dataset):
    df = summary.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    cn_df = df[df["selector"] == "DyGraFS"]
    non_cn_df = df[df["selector"] != "DyGraFS"]

    for ax, similarity_function in zip(axes, SIMILARITY_FUNCTIONS):

        # Non-CN selectors (independent of Função de similaridade)
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

        # CN selectors for this Função de similaridade
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

        ax.set_title(f"Função de similaridade: {similarity_function}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_xlabel("Número médio de características selecionadas")
    axes[3].set_xlabel("Número médio de características selecionadas")

    # De-duplicate legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4)

    fig.suptitle(f"{dataset.replace('four_class_nsclc', 'nsclc_quatro_classes')}: Acurácia balanceada vs Características por Função de Similaridade", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_features_by_similarity_function_pt.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def accuracy_vs_features_by_threshold_pt(summary, dataset):
    df = summary.copy()

    bins = [-np.inf, 0.15, 0.30, 0.60, np.inf]
    labels = [
        "limiar ≤ 0.15",
        "0.15 < limiar ≤ 0.30",
        "0.30 < limiar ≤ 0.60",
        "limiar > 0.60"
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

    axes[0].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_xlabel("Número médio de características selecionadas")
    axes[3].set_xlabel("Número médio de características selecionadas")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4)

    fig.suptitle(f"{dataset.replace('four_class_nsclc', 'nsclc_quatro_classes')}: Acurácia balanceada vs Características por Intervalo de limiar", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_features_by_threshold_pt.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def accuracy_vs_features_by_cn_selector_pt(summary, dataset):
    """
    Balanced Acurácia balanceada vs Número médio de características selecionadas,
    separated por CN selector (one subplot per CN selector)
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

        ax.set_title(f"Seletor de rede: {cn_sel}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_xlabel("Número médio de características selecionadas")
    axes[3].set_xlabel("Número médio de características selecionadas")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4)

    fig.suptitle(f"{dataset.replace('four_class_nsclc', 'nsclc_quatro_classes')}: Acurácia balanceada vs Características por Seletor de rede", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_features_by_cn_selector_pt.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def accuracy_vs_threshold_by_cn_selector_pt(summary, dataset):
    """
    Generates 4 subplots (one for each CN selector) showing 
    Balanced Acurácia balanceada vs. Threshold.
    """
    df = summary.copy()

    # Create the 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    # Iterate through selectors (assuming CN_SELECTORS is a predefined list of 4)
    for ax, cn_sel in zip(axes, sorted(CN_SELECTORS)):

        # 1. Plot "Outros seletores" as background reference (Steelblue)
        non_cn = df[~df["cn_selector"].isin(CN_SELECTORS)]
        ax.scatter(
            non_cn["threshold"],
            non_cn["balanced_accuracy_mean"],
            c="steelblue",
            s=60,
            alpha=0.4, # Slightly more transparent to emphasize the target
            label="Outros seletores"
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

        ax.set_title(f"Seletor de rede: {cn_sel}")
        ax.grid(alpha=0.3)

    # Add axis labels to the outer plots
    axes[0].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_ylabel("Acurácia balanceada (média)")
    axes[2].set_xlabel("Limiar")
    axes[3].set_xlabel("Limiar")

    # Handle the Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    # Main Title and Layout
    fig.suptitle(f"{dataset.replace('four_class_nsclc', 'nsclc_quatro_classes')}: Acurácia balanceada vs. Limiar por Seletor de rede", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the output
    out_path = f"outputs/{dataset}/{dataset}_accuracy_vs_threshold_by_cn_selector_pt.png"
    plt.savefig(out_path, dpi=300)
    plt.close()