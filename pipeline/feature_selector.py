import pandas as pd
import networkx as nx
from pipeline.build_network import generate_network
import numpy as np

def select_community_centers(
    csv_path,
    threshold=0.7,
    mapping_file="feature_mapping.csv",
    png_file="radiomic_graph.png",
    method="lp",  # "lp" (Label Propagation) as default
    check_eigen=False,  # If True, use eigenvector centrality to refine selection
    dataset="brats_africa"
):
    """
    Detect feature communities in a prebuilt network and return center features.

    Args:
        csv_path (str): Path to feature CSV.
        threshold (float): Correlation threshold for edge creation.
        mapping_file (str): Mapping file for feature names.
        png_file (str): Output graph visualization path.
        method (str): Community detection/selection method ("lp" or "pr").
        check_eigen (bool): Whether to use eigenvector centrality for refinement.

    Returns:
        centers (list): Selected feature names (community centers).
    """
    # Build or load network
    G = generate_network(csv_path, threshold, mapping_file, png_file, dataset)
    print("Initial number of features:", G.number_of_nodes())

    # Detecting communities
    if method == "lp":
        communities = list(nx.algorithms.community.label_propagation_communities(G))
    elif method == "pr":
        # PageRank can be used globally; we can group nodes afterward by high correlation
        pr_scores = nx.pagerank(G)
        # Create artificial "communities" by thresholding PR values into quantiles
        quantiles = np.percentile(list(pr_scores.values()), [25, 50, 75])
        communities = [set() for _ in range(4)]
        for node, score in pr_scores.items():
            if score <= quantiles[0]:
                communities[0].add(node)
            elif score <= quantiles[1]:
                communities[1].add(node)
            elif score <= quantiles[2]:
                communities[2].add(node)
            else:
                communities[3].add(node)
    else:
        raise ValueError("Invalid method. Choose 'lp' for Label Propagation or 'pr' for PageRank.")

    centers = []
    for community in communities:
        if len(community) == 0:
            continue
        subgraph = G.subgraph(community)

        # Optional eigenvalue-based refinement
        if check_eigen:
            try:
                eig_centrality = nx.eigenvector_centrality(subgraph, max_iter=1000)
                center = max(eig_centrality, key=eig_centrality.get)
            except nx.PowerIterationFailedConvergence:
                print("Eigenvector centrality failed to converge; falling back to degree centrality.")
                center = max(subgraph.degree, key=lambda x: x[1])[0]
        else:
            # Default center selection method
            if method == "pr":
                pr_sub = nx.pagerank(subgraph)
                center = max(pr_sub, key=pr_sub.get)
            else:
                center = max(subgraph.degree, key=lambda x: x[1])[0]

        centers.append(center)

    print("Selected features:", centers)
    return centers
