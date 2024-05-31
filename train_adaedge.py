from baseline import train_wrapper, GCN_Net
import torch
from torch_geometric.utils import to_networkx, from_networkx
from itertools import combinations
import networkx as nx
from tqdm import tqdm
import math


def inference(data, model):
    model.eval()
    with torch.no_grad():
        x, label, edge_index, val_mask, train_mask = data.x, data.y, data.edge_index, data.val_mask, data.train_mask
        pred = model(x, edge_index).detach()
        outcome = torch.argmax(pred, dim=1)

    return pred, outcome


def adjust_graph_topology(data, model_path='model.pt', threshold=0.15, edge_to_remove=100):
    """
        Input:
            data: torch_geometric.data.Data
            model_path: str
    """
    threshold = math.log(threshold)  # The predicted result is derived from log_softmax

    model = GCN_Net(2, data.num_features, 32, 7, 0.4)  # NOTE: cannot change
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data.to(device)
    prob, preds = inference(data, model)
    confidences = torch.max(prob, dim=1).values

    # Adjust the graph topology
    G = to_networkx(data, to_undirected=True)
    train_nodes = data.train_mask.nonzero(as_tuple=True)[0].tolist()  # Get list of training node indices
    all_nodes = list(G.nodes)

    # Track changes
    edge_changes = False
    edges_removed = 0

    bar = tqdm(combinations(all_nodes, 2), total=len(all_nodes) * (len(all_nodes) - 1) // 2)
    for u, v in bar:
        if preds[u] != preds[v] and G.has_edge(u, v) and confidences[u] > threshold and confidences[v] > threshold:
            G.remove_edge(u, v)
            edges_removed += 1
            edge_changes = True

        if edges_removed == edge_to_remove:
            break

    if edges_removed != edge_to_remove:
        assert False, f"Not enough edges to remove, got {edges_removed} edges, expected {edge_to_remove} edges."

    # Only update edge_index if there were changes
    if edge_changes:
        new_edge_index = from_networkx(G).edge_index
        data.edge_index = new_edge_index

    return data.edge_index


if __name__ == "__main__":
    import os

    data = torch.load('data\data.pt')
    model = GCN_Net(2, data.num_features, 32, 7, 0.4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data.to(device)

    if not os.path.exists('ada_model.pt'):
        train_wrapper(data,
                      model,
                      max_epoches=800,
                      lr=0.0002,
                      weight_decay=0.0005,
                      eval_interval=20,
                      early_stopping=100,
                      use_val=True,
                      save_path=f'ada_model.pt')

    updated_edges = adjust_graph_topology(data, model_path='ada_model.pt', threshold=0.15, edge_to_remove=600)
