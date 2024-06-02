from baseline import train_wrapper, GCN_Net
import torch
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from tqdm import tqdm
import numpy as np
import heapq
device = "mps"

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

def f_GPU(A, L):
    # Convert input matrix to PyTorch Tensor and move it to the GPU
    A = A.astype(np.float32) # mps only supports float32
    A_tensor = torch.from_numpy(A).to(device=device)
    I_tensor = torch.eye(A_tensor.shape[0], device=device)
    # Perform the matrix operations on the GPU
    A_squared_tensor = torch.matmul(A_tensor, A_tensor)
    result_tensor = A_squared_tensor + 2 * A_tensor + I_tensor
    result_tensor = torch.matmul(result_tensor, L)
    # Convert the result back to a NumPy ndarray
    return result_tensor

def I(A, L):
    L = L.float()
    fAl = f_GPU(A, L)
    norms_f = torch.norm(fAl)
    norms_L = torch.norm(L)
    IA = torch.matmul(L, fAl.t()) / (norms_L * norms_f)
    return IA

def topoinf(A, u, v, labels): #topoinf for edge e_{uv}
    A_ = A.copy()
    A_[u][v] = 0
    A_[v][u] = 0
    return I(A_, labels)

def top_n_edges(A, n, labels):
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
    bar = tqdm(get_all_edges(A))
    for u, v in bar:
        topoinfuv = topoinf(A, u, v, labels)
        if len(min_heap) < n:
            heapq.heappush(min_heap, (u,v,topoinfuv))
        else:
        # If the current edge has a higher score than the smallest in the heap
            if topoinfuv > min_heap[0][2]:
                # Replace the smallest element in the heap with the current edge
                heapq.heapreplace(min_heap, (u,v,topoinfuv))
    # The heap contains the top n edges
    return [(u,v) for u,v,_ in min_heap]



def adjust_graph_topology_topoinf_easy(data, model_path='model.pt', edge_to_remove=100):
    """
        Input:
            data: torch_geometric.data.Data
            model_path: str
    """
    model = GCN_Net(2, data.num_features, 32, 7, 0.4)  # NOTE: cannot change
    model.load_state_dict(torch.load(model_path))
    device = torch.device("mps")
    model.to(device)
    data.to(device)
    # Adjust the graph topology
    labels = data.y
    G = to_networkx(data, to_undirected=True)
    adj_t = nx.to_numpy_array(G)
    edges_to_remove = top_n_edges(adj_t, edge_to_remove, labels)
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

    updated_edges = adjust_graph_topology(data, model_path='ada_model.pt',edge_to_remove=600, lambda_ = 0.1)