from baseline import train_wrapper, GCN_Net
import torch
from torch_geometric.utils import to_networkx, from_networkx
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
import numpy as np
import heapq
device = "mps"

def to_onehot(L):
    unique_categories = torch.unique(L)
    num_categories = unique_categories.size(0)  # Number of unique categories
    one_hot_labels = F.one_hot(L, num_classes=num_categories)
    return one_hot_labels

def inference(data, model):
    model.eval()
    with torch.no_grad():
        x, label, edge_index, val_mask, train_mask = data.x, data.y, data.edge_index, data.val_mask, data.train_mask
        pred = model(x, edge_index).detach()
        non_zero_indices = torch.nonzero(label, as_tuple=True)[0]
        pred = torch.exp(pred)
        # print(non_zero_indices)
        # print(label[non_zero_indices])
        label = to_onehot(label)
        # print(label.shape)
        for i in non_zero_indices:
            pred[i] = label[i]
        
    return pred

def get_all_edges(A):
    """
    Get all edges from the adjacency matrix of an undirected graph.

    Parameters:
    A (np.ndarray): The adjacency matrix of the graph.

    Returns:
    list: A list containing tuples representing all edges in the graph.
    """
    num_nodes = A.shape[0]
    edges = []

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if A[i, j] != 0:
                edges.append((i, j))

    return edges

"""
def f_GPU(A_tensor, L):
    
    I_tensor = torch.eye(A_tensor.shape[0], device=device)
    
    A_squared_tensor = torch.matmul(A_tensor, A_tensor)
    result_tensor = A_squared_tensor + 2 * A_tensor + I_tensor # we are using two layer GCN, so K = 2, A^k = A^2
    row_sums = result_tensor.sum(dim=1, keepdim=True)
    result_tensor = result_tensor / row_sums
    result_tensor = torch.matmul(result_tensor, L)
    # Convert the result back to a NumPy ndarray
    return result_tensor
"""

def f_GPU(A, L, K = 2):
    A_tensor = A.to(device) 
    A_K = torch.pow(A_tensor, K)
    row_sum = torch.sum(A_K, dim=1, keepdim=True)
    A_K = A_K / row_sum
    result_tensor = torch.matmul(A_K, L)
    return result_tensor

def I(A, L, similarity_measure='cos'):
    L = L.float()
    fAl = f_GPU(A, L)
    
    if similarity_measure == 'cos':
        IA = F.cosine_similarity(L, fAl, dim=-1).mean()
    elif similarity_measure == 'kl':
        IA = - F.kl_div(F.log_softmax(fAl, dim=-1), L, reduction='batchmean')
    elif similarity_measure == 'euc':
        IA = - torch.norm(L - fAl, dim=-1).mean()  
    else:
        raise ValueError(f"Unknown similarity measure: {similarity_measure}")
    
    return IA

def topoinf(A, u, v, degrees, labels, lambda_, default_inf = -1, sim_metric = "cos"): #topoinf for edge e_{uv}
    # A is raw adj matrix
    A = torch.from_numpy(A).float().to(device)
    degrees = torch.from_numpy(degrees).float().to(device)
    labels.to(device)
    
    A_ = A.clone()
    A_[u][v] = 0
    A_[v][u] = 0
    du = 1/(degrees[u]) - 1/(degrees[u]-1) if degrees[u] != 1 else default_inf
    dv = 1/(degrees[v]) - 1/(degrees[v]-1) if degrees[v] != 1 else default_inf
    
    # A^tilde = A + I, D^tilde = D + I, A^hat = D^tilde(-1/2) A^tilde D^tilde(-1/2)
    D = torch.diag(degrees)
    D_d = D + torch.eye(D.shape[0], device=device)
    
    A_d = A_ + torch.eye(A_.shape[0], device=device)
    D_d_inv_sqrt = torch.diag(1/torch.sqrt(D_d.sum(dim=1)))
    
    A_hat = torch.matmul(torch.matmul(D_d_inv_sqrt, A_d), D_d_inv_sqrt)

    return I(A_hat, labels, similarity_measure=sim_metric) + lambda_ * (du+dv)

def top_n_edges(A, n, degrees, labels, lambda_, default_inf = -1, sim_metric = "cos"):
    """
    Find the top n edges with the highest scores.

    Parameters:
    edges (list of Edge): List of edges where each edge has a score.
    n (int): Number of top edges to find.

    Returns:
    list of Edge: List of top n edges with the highest scores.
    """
    # Use a min-heap to store the top n edges
    min_heap = []
    bar = tqdm(get_all_edges(A), desc="Calculating topoinf, lambda = {}, default_inf = {}, sim = {}".format(lambda_, default_inf, sim_metric), ncols=100)
    for u, v in bar:
        topoinfuv = topoinf(A, u, v, degrees, labels, lambda_, default_inf, sim_metric = sim_metric)
        if len(min_heap) < n:
            heapq.heappush(min_heap, (u,v,topoinfuv))
        else:
        # If the current edge has a higher score than the smallest in the heap
            if topoinfuv > min_heap[0][2]:
                # Replace the smallest element in the heap with the current edge
                heapq.heapreplace(min_heap, (u,v,topoinfuv))
    # The heap contains the top n edges
    return [(u,v) for u,v,_ in min_heap]



def adjust_graph_topology_topoinf_easy(data, model, edge_to_remove=100, lambda_ = 0.1, default_inf = -1, sim_metric = "cos"):
    """
        Input:
            data: torch_geometric.data.Data
            model_path: str
    """
    device = torch.device("mps")
    data.to(device)
    # Adjust the graph topology
    preds = inference(data, model)
    G = to_networkx(data, to_undirected=True)
    adj_t = nx.to_numpy_array(G)
    degrees = np.sum(adj_t, axis=1)
    edges_to_remove = top_n_edges(adj_t, edge_to_remove, degrees, preds, lambda_, default_inf, sim_metric = sim_metric)
    for u, v in edges_to_remove:
        G.remove_edge(u,v)
    # Only update edge_index if there were changes
    new_edge_index = from_networkx(G).edge_index

    # print(data.edge_index)
    # print(new_edge_index)

    # original_edge_index = data.edge_index
    # original_edge_index = set([tuple(list(edge.numpy())) for edge in original_edge_index.T])
    # edges = set([tuple(list(edge.numpy())) for edge in new_edge_index.T])
    # print(list(original_edge_index)[:30])
    # print(list(edges)[:30])
    # targ_edges = list(original_edge_index - edges)
    # print(len(targ_edges))

    return new_edge_index


if __name__ == "__main__":
    import os

    data = torch.load(os.path.join('data','data.pt'))
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

    updated_edges = adjust_graph_topology_topoinf_easy(data, model_path='ada_model.pt',edge_to_remove=600, lambda_ = 0.1)
