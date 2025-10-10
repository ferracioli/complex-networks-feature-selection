import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def generate_network(csv_path, threshold=0.7, mapping_file="feature_mapping.csv", png_file="radiomic_graph.png"):
    """
    Generate a similarity network of radiomic features and save graph as PNG.
    
    Args:
        csv_path (str): Path to CSV file.
        threshold (float): Similarity threshold to create edges.
        mapping_file (str): File where the mapping of nodes to columns will be saved.
        png_file (str): Output file for PNG of network.
    
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

    # --- Save mapping of node index to feature name ---
    mapping_df = pd.DataFrame({"node": range(len(feature_names)), "feature": feature_names})
    mapping_df.to_csv(mapping_file, index=False)

    # --- Compute cosine similarity ---
    similarity_matrix = cosine_similarity(feature_vectors)

    # --- Build graph ---
    G = nx.Graph()
    for feat in feature_names:
        G.add_node(feat)

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
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
    plt.title("Radiomic Feature Similarity Graph", fontsize=14)
    plt.tight_layout()
    plt.savefig(png_file, dpi=300)
    plt.close()

    return G

# generate_network("radiomic_features_brats_africa.csv", threshold=0.7, mapping_file="brats_africa_feature_mapping.csv", png_file="brats_africa_radiomic_graph.png")

# Version 1
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity

# def generate_network():
#     # --- Load data ---
#     df = pd.read_csv("radiomics.csv")

#     # Drop target and non-feature columns
#     drop_cols = ["glioma", "exam_path", "gt_path", "patient_id"]
#     features_df = df.drop(columns=drop_cols, errors="ignore")

#     # --- Transpose so rows = features, cols = patients ---
#     # Each feature is represented by a vector across patients
#     feature_vectors = features_df.T  

#     # --- Compute cosine similarity between features ---
#     similarity_matrix = cosine_similarity(feature_vectors)

#     # Feature names
#     feature_names = feature_vectors.index.tolist()

#     # --- Build graph ---
#     G = nx.Graph()

#     # Add nodes
#     for feat in feature_names:
#         G.add_node(feat)

#     # Add edges if similarity is above threshold (e.g., > 0.7)
#     threshold = 0.7
#     for i in range(len(feature_names)):
#         for j in range(i+1, len(feature_names)):
#             sim = similarity_matrix[i, j]
#             if sim > threshold:
#                 G.add_edge(feature_names[i], feature_names[j], weight=sim)

#     # --- Plot graph ---
#     plt.figure(figsize=(12, 10))
#     pos = nx.spring_layout(G, seed=42)  # layout
#     nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue", alpha=0.9)
#     nx.draw_networkx_edges(G, pos, width=1, alpha=0.6)
#     nx.draw_networkx_labels(G, pos, font_size=8)

#     # Optional: draw edge weights
#     # edge_labels = nx.get_edge_attributes(G, 'weight')
#     # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

#     plt.axis("off")
#     plt.title("Radiomic Feature Similarity Graph", fontsize=14)
#     plt.tight_layout()
#     plt.savefig("radiomic_graph.png", dpi=300)
#     plt.show()

#     return G
