from baseline import inference_wrapper
import torch
import torch_geometric.utils as pyg_utils
import networkx as nx

def edge_level_augmentation(graph, pe, pt, centrality_measure='degree'):
    graph_aug = graph.clone()
    pt = torch.tensor(pt)
    
    # Calculate node centrality measures
    if centrality_measure == 'degree':
        node_centrality = pyg_utils.degree(graph_aug.edge_index[0], graph_aug.num_nodes)
    elif centrality_measure == 'eigenvector':
        G = nx.Graph()
        G.add_edges_from(graph_aug.edge_index.t().numpy().tolist())
        node_centrality = nx.eigenvector_centrality(G)
        node_centrality = torch.tensor([node_centrality[i] for i in range(graph_aug.num_nodes)])
    else:
        G = nx.Graph()
        G.add_edges_from(graph_aug.edge_index.t().numpy().tolist())
        node_centrality = nx.pagerank(G)
        node_centrality = torch.tensor([node_centrality[i] for i in range(graph_aug.num_nodes)])
    
    # Calculate edge centrality measures
    edge_weights = (node_centrality[graph_aug.edge_index[0]] + node_centrality[graph_aug.edge_index[1]]) / 2.0
    
    # Take the logarithm of the edge weights
    edge_weights = torch.log(edge_weights)
    
    # Calculate the maximum and average edge weights
    max_weight = edge_weights.max().item()
    avg_weight = edge_weights.mean().item()
    
    # Calculate the probability of removing each edge
    edge_probabilities = torch.min(((max_weight - edge_weights) /
                                    (max_weight - avg_weight)) * pe, pt)
    
    # Remove edges based on their probabilities
    mask = torch.rand(edge_probabilities.size()) < edge_probabilities
    removed_edge_indices = graph_aug.edge_index[:, mask]
    removed_edge_count = removed_edge_indices.size(1)
    graph_aug.edge_index = graph_aug.edge_index[:, ~mask]
    
    # Print the number of removed edges
    print(f"Number of removed edges: {removed_edge_count}")
    
    return graph_aug

raw_data = torch.load("data\data.pt")
augmented_data = edge_level_augmentation(raw_data, 0.15, 0.2, centrality_measure='degree')

original_result = inference_wrapper(raw_data, model_path='model.pt')
print(original_result)

aug_result = inference_wrapper(augmented_data, model_path='model.pt')
print(aug_result)