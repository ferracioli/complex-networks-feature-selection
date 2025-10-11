import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import spearmanr, pearsonr

def generate_network(
    csv_path,
    threshold=0.7,
    mapping_file="feature_mapping.csv",
    png_file="radiomic_graph.png",
    link_method="cosine"
):
    """
    Generate a similarity network of radiomic features and save graph as PNG.
    
    Args:
        csv_path (str): Path to CSV file.
        threshold (float): Similarity threshold to create edges.
        mapping_file (str): File where the mapping of nodes to columns will be saved.
        png_file (str): Output file for PNG of network.
        link_method (str): Method to compute feature similarity.
            Options: ['cosine', 'spearman', 'pearson', 'euclidean']
    
    Returns:
        G (networkx.Graph): Graph of radiomic features.
    """
    # --- Load data ---
    df = pd.read_csv(csv_path)

    # Drop target and non-feature columns
    drop_cols = ["glioma", "exam_path", "gt_path", "patient_id"]
    features_df = df.drop(columns=drop_cols, errors="ignore")

    # --- Transpose so rows = features, cols = patients ---
    feature_vectors = features_df.T
    feature_names = feature_vectors.index.tolist()
    n_features = len(feature_names)

    # --- Save mapping of node index to feature name ---
    mapping_df = pd.DataFrame({"node": range(n_features), "feature": feature_names})
    mapping_df.to_csv(f"outputs/brats_africa/{mapping_file}", index=False)

    # --- Compute similarity matrix based on chosen method ---
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

    else:
        raise ValueError("Invalid link_method. Choose from ['cosine', 'spearman', 'pearson', 'euclidean']")

    # --- Build graph ---
    G = nx.Graph()
    for feat in feature_names:
        G.add_node(feat)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            sim = similarity_matrix[i, j]
            if sim > threshold:
                G.add_edge(feature_names[i], feature_names[j], weight=sim)

    # --- Save PNG of graph ---
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis("off")
    plt.title(f"Radiomic Feature Graph ({link_method} similarity)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"outputs/brats_africa/{link_method}_{png_file}", dpi=300)
    plt.close()

    return G


# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity

# def generate_network(csv_path, threshold=0.7, mapping_file="feature_mapping.csv", png_file="radiomic_graph.png"):
#     """
#     Generate a similarity network of radiomic features and save graph as PNG.
    
#     Args:
#         csv_path (str): Path to CSV file.
#         threshold (float): Similarity threshold to create edges.
#         mapping_file (str): File where the mapping of nodes to columns will be saved.
#         png_file (str): Output file for PNG of network.
    
#     Returns:
#         G (networkx.Graph): Graph of radiomic features.
#     """
#     # --- Load data ---
#     df = pd.read_csv(csv_path)

#     # Drop target and non-feature columns
#     drop_cols = ["glioma", "exam_path", "gt_path", "patient_id"]
#     features_df = df.drop(columns=drop_cols, errors="ignore")

#     # --- Transpose so rows = features, cols = patients ---
#     feature_vectors = features_df.T
#     feature_names = feature_vectors.index.tolist()

#     # --- Save mapping of node index to feature name ---
#     mapping_df = pd.DataFrame({"node": range(len(feature_names)), "feature": feature_names})
#     mapping_df.to_csv(mapping_file, index=False)

#     # --- Compute cosine similarity ---
#     similarity_matrix = cosine_similarity(feature_vectors)

#     # --- Build graph ---
#     G = nx.Graph()
#     for feat in feature_names:
#         G.add_node(feat)

#     for i in range(len(feature_names)):
#         for j in range(i + 1, len(feature_names)):
#             sim = similarity_matrix[i, j]
#             if sim > threshold:
#                 G.add_edge(feature_names[i], feature_names[j], weight=sim)

#     # --- Save PNG of graph ---
#     plt.figure(figsize=(12, 10))
#     pos = nx.spring_layout(G, seed=42)
#     nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue", alpha=0.9)
#     nx.draw_networkx_edges(G, pos, width=1, alpha=0.6)
#     nx.draw_networkx_labels(G, pos, font_size=8)
#     plt.axis("off")
#     plt.title("Radiomic Feature Similarity Graph", fontsize=14)
#     plt.tight_layout()
#     plt.savefig(png_file, dpi=300)
#     plt.close()

#     return G