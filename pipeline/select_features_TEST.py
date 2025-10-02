import pandas as pd
import networkx as nx
from network_builder import generate_network

def select_community_centers(csv_path, threshold=0.7):
    """
    Detect communities and select center features (highest degree).
    
    Args:
        csv_path (str): Path to CSV file.
        threshold (float): Edge creation threshold.
    
    Returns:
        centers (list): Selected feature names (community centers).
    """
    # Build graph
    G = generate_network(csv_path, threshold)

    # --- Community detection with label propagation ---
    communities = list(nx.algorithms.community.label_propagation_communities(G))

    centers = []
    for community in communities:
        subgraph = G.subgraph(community)
        # Select node with maximum degree
        center = max(subgraph.degree, key=lambda x: x[1])[0]
        centers.append(center)

    return centers

def build_reduced_csv(csv_path, output_csv="reduced_radiomics.csv", threshold=0.7):
    """
    Build a reduced CSV with only selected features (community centers) + target + paths.
    
    Args:
        csv_path (str): Path to input CSV.
        output_csv (str): Path to save reduced CSV.
        threshold (float): Threshold for edge creation in graph.
    
    Returns:
        reduced_df (pd.DataFrame): Reduced dataframe.
    """
    centers = select_community_centers(csv_path, threshold)
    df = pd.read_csv(csv_path)

    keep_cols = centers + ["glioma", "exam_path", "gt_path", "patient_id"]
    reduced_df = df[keep_cols]

    reduced_df.to_csv(output_csv, index=False)
    return reduced_df

if __name__ == "__main__":
    csv_file = "radiomics.csv"  # adjust path
    selected_features = select_community_centers(csv_file, threshold=0.7)
    print("Selected community centers (features):")
    for feat in selected_features:
        print(" -", feat)

    reduced_df = build_reduced_csv(csv_file, "reduced_radiomics.csv", threshold=0.7)
    print(f"\nReduced CSV saved with {reduced_df.shape[1]} columns and {reduced_df.shape[0]} rows.")
