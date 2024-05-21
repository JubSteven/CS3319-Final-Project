import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import time
import copy
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import networkx as nx
import community as cm


def get_scores(edges_pos, edges_neg, A_pred, adj_label):
    # get logits and labels
    preds_pos = A_pred[edges_pos[:, 0], edges_pos[:, 1]]
    preds_neg = A_pred[edges_neg[:, 0], edges_neg[:, 1]]

    logits = np.hstack([preds_pos, preds_neg])
    labels = np.hstack([np.ones(preds_pos.size(0)), np.zeros(preds_neg.size(0))])

    roc_auc = roc_auc_score(labels, logits)
    ap_score = average_precision_score(labels, logits)
    precisions, recalls, thresholds = precision_recall_curve(labels, logits)
    pr_auc = auc(recalls, precisions)

    f1s = np.nan_to_num(2 * precisions * recalls / (precisions + recalls))
    best_comb = np.argmax(f1s)
    f1 = f1s[best_comb]
    pre = precisions[best_comb]
    rec = recalls[best_comb]
    thresh = thresholds[best_comb]

    adj_rec = copy.deepcopy(A_pred)
    adj_rec[adj_rec < thresh] = 0
    adj_rec[adj_rec >= thresh] = 1

    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = adj_rec.view(-1).long()
    recon_acc = (preds_all == labels_all).sum().float() / labels_all.size(0)
    results = {
        'roc': roc_auc,
        'pr': pr_auc,
        'ap': ap_score,
        'pre': pre,
        'rec': rec,
        'f1': f1,
        'acc': recon_acc,
        'adj_recon': adj_rec
    }
    return results


def sample_graph_det(adj_orig, A_pred, remove_edge_num=100):
    if remove_edge_num == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    edges = np.asarray(orig_upper.nonzero()).T
    if remove_edge_num:
        n_remove = remove_edge_num
        edge_prob = A_pred[edges.T[0], edges.T[1]]
        edge_index_to_remove = np.argpartition(edge_prob, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[edge_index_to_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    # Recover the edges to [a,b] and [b,a] instead of [a,b]
    edges_pred = np.concatenate([edges_pred, edges_pred[:, ::-1]])

    return edges_pred


def sample_graph_community(adj_orig, A_pred, remove_edge_num=100, tau=0.1):
    if remove_edge_num == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    edges = np.asarray(orig_upper.nonzero()).T
    partition = cm.best_partition(nx.Graph(adj_orig), random_state=42)

    if remove_edge_num:
        n_remove = remove_edge_num
        # Create a list storing the current weight of each node
        node_count = np.zeros(len(partition))
        edge_index_to_remove = []
        for _ in range(n_remove):
            # Find the edge with the smallest probability
            edge_prob = A_pred[edges.T[0], edges.T[1]]
            remaining_edges = [idx for idx in np.argsort(edge_prob) if idx not in edge_index_to_remove]
            if len(remaining_edges) == 0:
                break
            min_edge_idx = remaining_edges[0]
            min_edge = edges[min_edge_idx]

            # Get the communities of the two nodes
            community_i = partition[min_edge[0]]
            community_j = partition[min_edge[1]]

            # Compute the penalty term
            penalty_i = node_count[community_i] / (2 * n_remove)
            penalty_j = node_count[community_j] / (2 * n_remove)

            # Update the rows i and j in the A_pred matrix with the penalty term
            A_pred[min_edge[0], :] *= (1 + penalty_i * tau)
            A_pred[min_edge[1], :] *= (1 + penalty_j * tau)

            # Add the edge index to the removal list
            edge_index_to_remove.append(min_edge_idx)

            # Update the node count for the communities
            node_count[community_i] += 1
            node_count[community_j] += 1

        # Remove the edges with the smallest probabilities
        edges_pred = np.delete(edges, edge_index_to_remove, axis=0)
    else:
        edges_pred = edges

    edges_pred = np.concatenate([edges_pred, edges_pred[:, ::-1]])

    return edges_pred


def louvain_clustering(adj, s_rec):
    """
    Performs community detection on a graph using the Louvain method
    :param adj: adjacency matrix of the graph
    :param s_rec: s hyperparameter for s-regular sparsification
    :return: adj_louvain, the Louvain community membership matrix obtained;
    nb_communities_louvain, the number of communities; partition, the community
    associated with each node from the graph
    """
    graph = nx.Graph(adj)

    # Community detection using the Louvain method
    partition = cm.best_partition(graph)
    communities_louvain = list(partition.values())

    # Number of communities found by the Louvain method
    nb_communities_louvain = np.max(communities_louvain) + 1

    # One-hot representation of communities
    communities_louvain_onehot = sp.csr_matrix(np.eye(nb_communities_louvain)[communities_louvain])

    # Community membership matrix (adj_louvain[i,j] = 1 if nodes i and j are in the same community)
    adj_louvain = communities_louvain_onehot.dot(communities_louvain_onehot.transpose())

    # Remove the diagonal
    adj_louvain = adj_louvain - sp.eye(adj_louvain.shape[0])

    # s-regular sparsification of adj_louvain
    adj_louvain = sparsification(adj_louvain, s_rec)

    return adj_louvain, nb_communities_louvain, partition


def sparsification(adj_louvain, s=1):
    """
    Performs an s-regular sparsification of the adj_louvain matrix (if possible)
    :param adj_louvain: the initial community membership matrix
    :param s: value of s for s-regular sparsification
    :return: s-sparsified adj_louvain matrix
    """

    # Number of nodes
    n = adj_louvain.shape[0]

    # Compute degrees
    degrees = np.sum(adj_louvain, axis=0).getA1()

    for i in range(n):

        # Get non-null neighbors of i
        edges = sp.find(adj_louvain[i, :])[1]

        # More than s neighbors? Subsample among those with degree > s
        if len(edges) > s:
            # Neighbors of i with degree > s
            high_degrees = np.where(degrees > s)
            edges_s = np.intersect1d(edges, high_degrees)
            # Keep s of them (if possible), randomly selected
            removed_edges = np.random.choice(edges_s, min(len(edges_s), len(edges) - s), replace=False)
            adj_louvain[i, removed_edges] = 0.0
            adj_louvain[removed_edges, i] = 0.0
            degrees[i] = s
            degrees[removed_edges] -= 1

    adj_louvain.eliminate_zeros()

    return adj_louvain


def get_basic_graph_features(adj, return_dict=False):
    """
    Node degree, node centrality, node clustering coefficient, shortest path length, Jacard similarity, Katz index
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.to_dense().numpy()
    else:
        adj = np.array(adj)

    # Convert the adjacency matrix to a NetworkX graph
    graph = nx.from_numpy_array(adj)

    degree = np.array(list(dict(graph.degree()).values()))  # Node degree

    centrality = np.array(list(nx.degree_centrality(graph).values()))  # Node centrality (using degree centrality)

    clustering_coefficient = np.array(list(nx.clustering(graph).values()))  # Node clustering coefficient

    # shortest_path_length = nx.floyd_warshall_numpy(graph)  # Shortest path length
    # shortest_path_length = np.array([[shortest_path_length[u][v] for v in graph.nodes()] for u in graph.nodes()])

    jacard_similarity = np.zeros(
        (len(graph.nodes()), len(graph.nodes())))  # Jacard similarity (assuming the graph is undirected)
    for u, v, j in nx.jaccard_coefficient(graph):
        jacard_similarity[u][v] = j
        jacard_similarity[v][u] = j

    katz_index = nx.katz_centrality_numpy(graph)  # TODO: perhaps wrong, not Katz index
    katz_index = np.array(list(katz_index.values()))

    # Create a dictionary containing the calculated features
    node_level_feats = {
        'degree': degree,
        'centrality': centrality,
        'clustering_coefficient': clustering_coefficient,
        # 'shortest_path_length': shortest_path_length,
        # 'jacard_similarity': jacard_similarity,
        'katz_index': katz_index
    }

    # TODO: add edge-level features
    edge_level_feats = {}
    # # Take the average of node_level_feats between the two nodes of an edge
    # for key in node_level_feats:
    #     edge_key = key + '_edge'
    #     edge_level_feats[edge_key] = np.array(
    #         [node_level_feats[key][u] + node_level_feats[key][v] for u, v in graph.edges()])
    #     edge_level_feats[edge_key] /= 2

    feats_dict = {**node_level_feats, **edge_level_feats}
    if return_dict:
        return feats_dict

    # TODO: merge the features into a single feature matrix, or design an NN to fuse the features
    features = np.concatenate([node_level_feats[key].reshape(-1, 1) for key in node_level_feats], axis=1)

    return features
