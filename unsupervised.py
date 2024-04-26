from baseline import inference_wrapper
import torch
import torch_geometric.utils as pyg_utils
import networkx as nx
from tqdm import tqdm


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
    edge_probabilities = torch.min(((max_weight - edge_weights) / (max_weight - avg_weight)) * pe, pt)

    # Remove edges based on their probabilities
    mask = torch.rand(edge_probabilities.size()) < edge_probabilities
    removed_edge_indices = graph_aug.edge_index[:, mask]
    removed_edge_count = removed_edge_indices.size(1)
    graph_aug.edge_index = graph_aug.edge_index[:, ~mask]

    # Print the number of removed edges
    # print(f"Number of removed edges: {removed_edge_count}")

    return graph_aug


def aug_eval(raw_data, sota=None):
    augmented_data = edge_level_augmentation(raw_data, 0.1, 0.2, centrality_measure='eigenvector')

    original_result = inference_wrapper(raw_data, model_path='model.pt')
    aug_result = inference_wrapper(augmented_data, model_path='model.pt')

    if sota is None:
        sota = original_result["val_acc"]

    elif aug_result["val_acc"] > sota:
        print("Update SOTA to : ", aug_result["val_acc"].item())
        print("Number of edges removed: ", raw_data.edge_index.size(1) - augmented_data.edge_index.size(1))
        torch.save(augmented_data, "data\data_augmented.pt")

    return original_result, aug_result


raw_data = torch.load("data\data.pt")
N = 1000
original_results = []
aug_results = []
sota = 0.7920  # This is aligned with the trained model.pt.

for i in tqdm(range(N)):
    original_result, aug_result = aug_eval(raw_data, sota)
    original_results.append(original_result["val_acc"].item())
    aug_results.append(aug_result["val_acc"].item())
    if aug_result["val_acc"].item() > sota:
        sota = aug_result["val_acc"].item()

# Plot the aug_results
import matplotlib.pyplot as plt

# Box plot
plt.boxplot([original_results, aug_results])
print(max(aug_results))
plt.show()
