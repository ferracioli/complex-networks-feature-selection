import community as community_louvain
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
np.random.seed(42)

# Loading the config json
with open('input/config.json', 'r') as file:
    config = json.load(file)

def generate_network(df, threshold=0.7, similarity_function="cosine"):
    """
    Generates a complex network based on the input dataframe and similarity function
    
    Args:
        df (radiomic DataFrame):
        threshold (float): Correlation threshold for edge creation.
        similarity_function (str): method used for edge generation in the network
        
    Returns:
        nx.Graph: The generated network in networkX format
    """

    # Keep non-feature columns to add back later
    non_feature_cols = ["glioma", "exam_path", "gt_path", "patient_id"]
    non_feature_cols = [c for c in non_feature_cols if c in df.columns]

    # Feature-only dataframe
    features_df = df.drop(columns=non_feature_cols, errors="ignore")

    # Drop features with very low variance(raise the value encountered in divide c /= stddev[None, :])
    # Keeping then results in a disjointed network
    low_var_thresh = 1e-6
    variances = features_df.var(axis=0)
    low_var_cols = variances[variances < low_var_thresh].index.tolist()
    if low_var_cols:
        print(f"Dropping {len(low_var_cols)} near-constant features.")
        features_df = features_df.drop(columns=low_var_cols)

    # Transpose so rows = features, cols = patients
    feature_vectors = features_df.T.values
    feature_names = features_df.columns.tolist()

    # Normalize features (important for cosine / Euclidean)
    feature_vectors = (feature_vectors - feature_vectors.mean(axis=1, keepdims=True)) / \
                      (feature_vectors.std(axis=1, keepdims=True) + 1e-8)

    # Compute similarity based on the selected param
    if similarity_function == "Cosine":
        similarity_matrix = cosine_similarity(feature_vectors)

    elif similarity_function == "Spearman":
        corr, _ = spearmanr(feature_vectors, axis=1)
        similarity_matrix = np.abs(np.triu(corr, 0))

    elif similarity_function == "Pearson":
        corr = np.corrcoef(feature_vectors)
        similarity_matrix = np.abs(corr)

    elif similarity_function == "Rho distance":
        corr = np.corrcoef(feature_vectors)
        corr = np.nan_to_num(corr, nan=0.0)
        d = np.sqrt(2 * (1 - corr))  # distance in [0, 2]
        # Convert to similarity in [0, 1]
        similarity_matrix = 1 - (d / np.max(d))

    else:
        raise ValueError(f"Invalid similarity_function: {similarity_function}")

    # Building  the graph
    G = nx.Graph()
    for feat in feature_names:
        G.add_node(feat)

    # Trying to optimize runtime with numpy by attributing only links higher than the thresh:
    idx_i, idx_j = np.where(np.triu(similarity_matrix, k=1) > threshold)

    edges = [
        (feature_names[i], feature_names[j], float(similarity_matrix[i, j]))
        for i, j in zip(idx_i, idx_j)
    ]
    G.add_weighted_edges_from(edges)

    # Add back non-feature columns
    features_df[non_feature_cols] = df[non_feature_cols]

    return G, features_df

def select_cn_centers(
    df,
    threshold=0.7,
    png_path="folder_name",
    cn_selector="Label Propagation", 
    similarity_function="Spearman",
    seed_nb=42,
    save_fig=False,
):
    """
    Detect feature communities in a prebuilt network and return nodes according to the selected method.

    Args:
        df (radiomic DataFrame):
        threshold (float): Correlation threshold for edge creation.
        png_path (str): Output graph visualization path.
        cn_selector (str): Community detection/selection method ("Label Propagation", "Louvain", "Betweeness" or "Page Rank").
        similarity_function (str): method used for edge generation in the network

    Returns:
        centers (list): Selected feature names (community centers).
    """
    # Build or load network
    vt = VarianceThreshold(threshold=0.001)
    df_reduced = vt.fit_transform(df)
    selected_cols = df.columns[vt.get_support()]
    df = pd.DataFrame(df_reduced, columns=selected_cols)

    G, _ = generate_network(df=df, threshold=threshold, similarity_function=similarity_function)
    if G.number_of_nodes() == 0:
        return []

    if cn_selector == "Label Propagation":

        communities = list(nx.algorithms.community.asyn_lpa_communities(G, seed=seed_nb))

        centers = []

        # --- Select one representative per community ---
        for comm in communities:
            # Single-node community -> keep it
            if len(comm) < 2:
                centers.extend(comm)
                continue

            # Induced subgraph for the community
            sub = G.subgraph(comm)

            # Degree centrality (linear-time, local)
            degrees = dict(sub.degree())

            # Select the community representative (degree + relevance to y when available)
            # center = _select_relevance_aware_center(sub, df, y, degrees)
            # original:
            center = max(degrees, key=degrees.get)
            centers.append(center)

    elif cn_selector == "Louvain":
        # Detect communities with Louvain
        partition = community_louvain.best_partition(G, random_state=seed_nb)

        # Organize nodes by community
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)

        centers = []

        # --- Select one representative per community ---
        for comm in communities.values():
            # Single-node community → keep it
            if len(comm) < 2:
                centers.extend(comm)
                continue

            # Induced subgraph for the community
            sub = G.subgraph(comm)

            # Degree centrality (linear-time, local)
            degrees = dict(sub.degree())

            # Select the community representative (degree + relevance to y when available)
            # center = _select_relevance_aware_center(sub, df, y, degrees)
            center = max(degrees, key=degrees.get)
            centers.append(center)

    # Page rank and betweenes were discarded in the experiment due to lower performance
    # but they can be used as well
    elif cn_selector == "Page Rank":
        pr = nx.pagerank(G)
        thr = np.percentile(list(pr.values()), 75)
        centers = [n for n, v in pr.items() if v >= thr]

    elif cn_selector == "Betweenness":
        btw = nx.betweenness_centrality(G)
        vals = np.array(list(btw.values()))
        z = (vals - vals.mean()) / vals.std()
        centers = [n for n, score in zip(G.nodes(), z) if score > 1.0]

    elif cn_selector == "Bridging Centrality":
        btw = nx.betweenness_centrality(G)
        deg = dict(G.degree())

        bridging = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0 or deg[node] == 0:
                bridging[node] = 0
                continue

            inv_deg_sum = sum(1 / deg[n] for n in neighbors if deg[n] > 0)
            coeff = (1 / deg[node]) * inv_deg_sum
            bridging[node] = btw[node] * coeff

        thr = np.percentile(list(bridging.values()), 75)
        centers = [n for n, v in bridging.items() if v >= thr]

    elif cn_selector == "Structural Diversity":
        # Calculating PageRank as a global centrality
        pr = nx.pagerank(G)

        # Sorting by most relevant nodes
        nodes_sorted = sorted(pr, key=pr.get, reverse=True)

        centers = []
        excluded = set()
        k = max(1, int(len(G) * 0.1)) 
        # Selects =~ the top 10% most diverse nodes

        for node in nodes_sorted:
            if node in excluded:
                continue
            centers.append(node)
            # Avoiding neigbors
            excluded.update(G.neighbors(node))
            excluded.add(node)
            if len(centers) >= k:
                break

    else:
        raise ValueError("Error: invalid method.")

    if save_fig:
        png_name = f"{str(threshold).replace('.', '')}_{similarity_function}_{cn_selector}_radiomic_graph.png"
        _save_graph_snapshot(G, centers, png_path, png_name.lower(), threshold, similarity_function, cn_selector, seed_nb)

    return centers


def _save_graph_snapshot(G, centers, png_path, png_file, threshold, similarity_function, cn_selector, seed_nb, dpi=70, figsize=(6, 5)):
    """
    Renders a low-resolution PNG of the full correlation graph with the
    selected community-center features highlighted in red (all other nodes in
    blue), so a run's parameter combination can be inspected visually. The
    parameter combination is expected to already be encoded in `png_path` by
    the caller (see `graph_selector` / `generate_dygrafs_diagnostic_plots`).

    Kept deliberately low-res (small figsize, dpi=70, no per-node labels,
    since with hundreds of radiomic features labels are unreadable anyway) so
    this is cheap enough to generate for every parameter combination tested,
    not just the single winning one.

    If a PNG already exists at `png_path`, generation is skipped: the same
    parameter combination is evaluated many times across CV folds/inner
    splits, and one representative snapshot per combination is the goal here,
    not one per fold.
    """
    file_path = f"{png_path}{png_file}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=seed_nb)

    node_colors = ["red" if n in centers else "skyblue" for n in G.nodes()]
    node_size = 40 if G.number_of_nodes() > 150 else 80
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.5)

    plt.title(f"Threshold: {threshold} | Similarity function: {similarity_function} | Method: {cn_selector}\nnodes={G.number_of_nodes()} | selected={len(centers)}", fontsize=9)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(file_path, dpi=dpi)
    plt.close()
