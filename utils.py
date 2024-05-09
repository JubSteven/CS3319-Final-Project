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
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        edge_index_to_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[edge_index_to_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    # Recover the edges to [a,b] and [b,a] instead of [a,b]
    edges_pred = np.concatenate([edges_pred, edges_pred[:, ::-1]])

    return edges_pred


# def sample_graph_ewm(adj_orig, A_pred, remove_edge_num=100):
#     basic_graph_feats = get_basic_graph_features(adj_orig, return_dict=True)
#     print(basic_graph_feats.keys())
#     print(basic_graph_feats['degree_edge'].shape)
#     assert False
#     # basic_graph_feats = (basic_graph_feats - basic_graph_feats.mean(axis=0)) / basic_graph_feats.std(axis=0)

#     orig_upper = sp.triu(adj_orig, 1)
#     edges = np.asarray(orig_upper.nonzero()).T

#     print(basic_graph_feats.shape)
#     print(A_pred.shape)
#     assert False


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
    # Take the average of node_level_feats between the two nodes of an edge
    for key in node_level_feats:
        edge_key = key + '_edge'
        edge_level_feats[edge_key] = np.array(
            [node_level_feats[key][u] + node_level_feats[key][v] for u, v in graph.edges()])
        edge_level_feats[edge_key] /= 2

    feats_dict = {**node_level_feats, **edge_level_feats}
    if return_dict:
        return feats_dict

    # TODO: merge the features into a single feature matrix, or design an NN to fuse the features
    features = np.concatenate([node_level_feats[key].reshape(-1, 1) for key in node_level_feats], axis=1)

    return features
