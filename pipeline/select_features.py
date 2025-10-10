import pandas as pd
import networkx as nx
from pipeline.build_network import generate_network

def select_community_centers(csv_path, threshold=0.7, mapping_file="feature_mapping.csv", png_file="radiomic_graph.png"):
    """
    Detect feature communities in a prebuilt network and return center features.

    Args:
        G (nx.Graph): Prebuilt feature correlation network.

    Returns:
        centers (list): Selected feature names (community centers).
    """
    G = generate_network(csv_path, threshold, mapping_file, png_file)

    print("Initial number of features:", G.number_of_nodes())

    # Detect communities via label propagation
    communities = list(nx.algorithms.community.label_propagation_communities(G))

    centers = []
    for community in communities:
        subgraph = G.subgraph(community)
        # Pick the node with highest degree as the community center
        center = max(subgraph.degree, key=lambda x: x[1])[0]
        centers.append(center)

    print("Selected features:", centers)
    return centers
