import pandas as pd
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import spearmanr, pearsonr

# Loading the config json
with open('input/config.json', 'r') as file:
    config = json.load(file)

def generate_network(
    csv_path,
    threshold=0.7,
    mapping_file="feature_mapping.csv",
    link_method="cosine",
    dataset="brats_africa"
):
    """
    Generate a similarity network of radiomic features and save graph as PNG.
    
    Args:
        csv_path (str): Path to CSV file.
        threshold (float): Similarity threshold to create edges.
        mapping_file (str): File where the mapping of nodes to columns will be saved.
        link_method (str): Method to compute feature similarity.
            Options: ['cosine', 'spearman', 'pearson', 'euclidean']
    
    Returns:
        G (networkx.Graph): Graph of radiomic features.
    """
    # Loading data
    df = pd.read_csv(csv_path)

    # Drop target and non-feature columns
    drop_cols = ["glioma", "exam_path", "gt_path", "patient_id"]
    features_df = df.drop(columns=drop_cols, errors="ignore")

    # Transpose so rows = features, cols = patients
    feature_vectors = features_df.T
    feature_names = feature_vectors.index.tolist()
    n_features = len(feature_names)

    # Save mapping of node index to feature name
    mapping_df = pd.DataFrame({"node": range(n_features), "feature": feature_names})
    mapping_df.to_csv(f"outputs/brats_africa/{mapping_file}", index=False)

    # Compute similarity matrix based on chosen method
    if link_method == "cosine":
        similarity_matrix = cosine_similarity(feature_vectors)

    elif link_method == "spearman":
        corr, _ = spearmanr(feature_vectors, axis=1)
        # Convert to similarity: higher when corr close to 1 or -1
        similarity_matrix = 1 - np.abs(1 - np.abs(corr))

    elif link_method == "pearson":
        corr = np.corrcoef(feature_vectors)
        similarity_matrix = np.abs(corr)

    elif link_method == "euclidean":
        distances = euclidean_distances(feature_vectors)
        # Convert to similarity (normalize to 0–1)
        similarity_matrix = 1 / (1 + distances)

    # Normalized angular distance
    elif link_method == "rho_distance":
        # Compute correlation matrix (Pearson)
        corr = np.corrcoef(feature_vectors)
        # Ensure no NaNs (can happen if constant columns exist)
        corr = np.nan_to_num(corr, nan=0.0)

        # Compute new distance metric: d = sqrt(1 - ρ / sqrt(2))
        d = np.sqrt(1 - corr / np.sqrt(2))

        # Convert to similarity (smaller distance = higher similarity)
        similarity_matrix = 1 / (1 + d)

    else:
        raise ValueError("Invalid link_method. Choose from ['cosine', 'spearman', 'pearson', 'euclidean']")

    # Building graph
    G = nx.Graph()
    for feat in feature_names:
        G.add_node(feat)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            sim = similarity_matrix[i, j]
            if sim > threshold:
                G.add_edge(feature_names[i], feature_names[j], weight=sim)

    # Saving png of graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis("off")
    plt.title(f"Radiomic Feature Graph ({link_method} similarity)", fontsize=14)
    plt.tight_layout()
    png_path = f"{config[dataset]["output_path"]}{dataset}_radiomic_graph.csv"
    plt.savefig(png_path, dpi=300)
    plt.close()

    return G
