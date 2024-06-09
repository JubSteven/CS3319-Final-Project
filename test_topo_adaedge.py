# train_adaedge
from baseline import train_wrapper, GCN_Net
import torch
from torch_geometric.utils import to_networkx, from_networkx
from itertools import combinations
import networkx as nx
from tqdm import tqdm
import math
import numpy as np

device = "mps"

from train_topoinf_easy import get_all_edges, topoinf

# GCN baseline
import numpy as np
import torch_geometric as pyg
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import time
from tqdm import tqdm
import copy

class GCN_Net(torch.nn.Module):

    def __init__(self, num_layers, num_features, hidden, num_classes, dropout):
        super(GCN_Net, self).__init__()

        self.num_layers = num_layers
        self.conv_list = torch.nn.ModuleList([])
        self.conv_list.append(GCNConv(num_features, hidden))
        for _ in range(self.num_layers - 2):
            self.conv_list.append(GCNConv(hidden, hidden))
        if num_layers >= 2:
            self.conv_list.append(GCNConv(hidden, num_classes))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)  # NOTE: there is a dropout layer.
        for i in range(self.num_layers - 1):
            x = self.conv_list[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x, edge_index)
        self.pred = F.log_softmax(x, dim=1)
        return self.pred

    def loss(self, label, mask):
        pred_loss = nn.NLLLoss(reduction='sum')(self.pred[mask], label[mask])
        return pred_loss


def train(data, model, opt, use_val=False):
    model.train()
    opt.zero_grad()
    x, label, edge_index, train_mask = data.x, data.y, data.edge_index, data.train_mask
    pred = model(x, edge_index)
    if not use_val:
        loss = model.loss(label, train_mask)
    else:
        # merge the train_mask and the val_mask (combine the 1s)
        merged_mask = torch.logical_or(train_mask, data.val_mask)
        loss = model.loss(label, merged_mask)
    loss.backward()
    opt.step()
    return loss


def val(data, model):
    model.eval()
    with torch.no_grad():
        x, label, edge_index, val_mask, train_mask = data.x, data.y, data.edge_index, data.val_mask, data.train_mask
        pred = model(x, edge_index).detach()
        outcome = (torch.argmax(pred, dim=1) == label)
        train_acc = torch.sum(outcome[train_mask]) / len(outcome[train_mask])
        val_acc = torch.sum(outcome[val_mask]) / len(outcome[val_mask])
    return {'train_acc': train_acc, 'val_acc': val_acc}


def print_eval_result(eval_result, prefix=''):
    if prefix:
        prefix = prefix + ' '
    print(f"{prefix}"
          f"Train Acc:{eval_result['train_acc']*100:6.2f} | "
          f"Val Acc:{eval_result['val_acc']*100:6.2f} | ")


def train_wrapper(data,
                  model,
                  max_epoches,
                  lr=0.001,
                  weight_decay=5e-4,
                  eval_interval=10,
                  early_stopping=100,
                  early_stopping_tolerance=1,
                  use_val=False,
                  save_path="models/model.pt"):
    model.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    val_acc_history = []
    start_time = time.time()
    bar = tqdm(range(1, max_epoches + 1))
    for epoch in bar:
        loss = train(data, model, opt, use_val=use_val)
        if epoch % eval_interval == 0:
            eval_result = val(data, model)
            bar.set_description(
                f'E: {epoch:3d} | L: {loss.item():.4f} | TR_ACC: {eval_result["train_acc"]:.4f} | VA_ACC: {eval_result["val_acc"]:.4f}'
            )
            if eval_result['val_acc'] > best_val_acc:
                best_val_acc = eval_result['val_acc']
                best_model_param = copy.deepcopy(model.state_dict())
            val_acc_history.append(eval_result['val_acc'])
            if early_stopping > 0 and len(val_acc_history) > early_stopping:
                mean_val_acc = torch.tensor(val_acc_history[-(early_stopping + 1):-1]).mean().item()
                if (eval_result['val_acc'] - mean_val_acc) * 100 < -early_stopping_tolerance:
                    print('[Early Stop Info] Stop at Epoch: ', epoch)
                    break
    model.load_state_dict(best_model_param)
    # Dump the best model
    torch.save(model.state_dict(), save_path)
    eval_result = val(data, model)

    return model


def inference_wrapper(data, model_path='model.pt'):
    """
        Input:
            data: torch_geometric.data.Data
            model_path: str
    """
    model = GCN_Net(2, data.num_features, 32, 7, 0.4)  # NOTE: cannot change
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data.to(device)
    results = val(data, model)
    return results


def get_central_candidates(candidates, NUM):
    total_candidates = len(candidates)

    start_index = (total_candidates - NUM) // 2
    end_index = start_index + NUM

    central_candidates = candidates[start_index:end_index]
    return central_candidates


def inference(data, model):
    model.eval()
    with torch.no_grad():
        x, label, edge_index, val_mask, train_mask = data.x, data.y, data.edge_index, data.val_mask, data.train_mask
        pred = model(x, edge_index).detach()
        outcome = torch.argmax(pred, dim=1)

    return pred, outcome

def adjust_graph_topology_opt(data, model, l_bound=0.15, u_bound = 0.8,edge_to_remove=100, use_topoinf=False, lambda_=1e-7, default_inf=-1):
    """
        Input:
            data: torch_geometric.data.Data
            model_path: str
    """
    data.to(device)
    G = to_networkx(data, to_undirected=True)
    adj_t = nx.to_numpy_array(G)
    edge_pairs = get_all_edges(adj_t)
    degrees = np.sum(adj_t, axis=1)
    bar = tqdm(edge_pairs, total=len(edge_pairs), desc="l_bound: {}, u_bound: {}, lambda: {}, default_inf: {}".format(l_bound, u_bound, lambda_, default_inf))
    l_bound = math.log(l_bound)  # The predicted result is derived from log_softmax
    u_bound = math.log(u_bound)
    

    model.to(device)

    prob, preds = inference(data, model)
    confidences = torch.max(prob, dim=1).values
    
    # Adjust the graph topology
    G = to_networkx(data, to_undirected=True)

    # Track changes
    edges_removed = 0
    candidates = []
    topo_infs = []
    for u, v in bar:
        if preds[u] != preds[v] and G.has_edge(u, v) and confidences[u] > l_bound and confidences[v] > l_bound and confidences[u] < u_bound and confidences[v] < u_bound:
            if use_topoinf:
                topo_inf = topoinf(adj_t, u, v, degrees, preds, lambda_=lambda_, default_inf=default_inf)
                topo_infs.append(topo_inf)
            else:
                topo_inf = 1
            # print(f"Topoinf: {topo_inf}")
            # print(f"Confidence: {confidences[u] + confidences[v]}")
            candidates.append((((confidences[u] + confidences[v]) + topo_inf) / 2 , u, v))
            # candidates.append((topo_inf, u, v))
    print(f"Got {len(candidates)} candidate edges.")

    if len(candidates) < edge_to_remove:
        assert False, f"Not enough edges to remove, got {len(candidates)} candidate edges, expected {edge_to_remove} edges."

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    
    # candidates = get_central_candidates(candidates, edge_to_remove)
    
    for _, u, v in candidates:
        G.remove_edge(u, v)
        edges_removed += 1

        if edges_removed == edge_to_remove:
            break

    # Only update edge_index if there were changes
    new_edge_index = from_networkx(G).edge_index

    return new_edge_index

import os
import pandas as pd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LAYER_NUM = 4
HIDDEN_DIM = 256
MODEL_PATH = f"models/Layer_{LAYER_NUM}_Hidden_{HIDDEN_DIM}/model_1.pt"

MAX_EPOCH = 800

LAYER_NUM = 2
HIDDEN_DIM = 32
MODEL_PATH = "/Users/husky/Desktop/GNN/CS3319-Final-Project/ada_model.pt"

EVAL_MODEL_PATHs = os.listdir("models/Layer_2_Hidden_32")

import time
stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

def from_adaedge(l_bound= 0.15, u_bound = 1, save_path = f"new_submission/adaedge_submission_{stamp}.csv", val=True, use_topoinf=False, lambda_ = 1e-7, default_inf=-1):
    torch.manual_seed(42)
    save_path = save_path + f"_l={l_bound}_u={u_bound}_topoinf={use_topoinf}_lambda={lambda_}_default_inf={default_inf}.csv"
    os.makedirs("submission", exist_ok=True)
    data_path = os.path.join("data", "data.pt")
    raw_data = torch.load(data_path)
    data = torch.load(data_path)
    
    model = GCN_Net(LAYER_NUM, raw_data.num_features, HIDDEN_DIM, 7, 0.4)
    if not os.path.exists(MODEL_PATH):
        train_wrapper(raw_data,
                      model,
                      max_epoches=800,
                      lr=0.0002,
                      weight_decay=0.0005,
                      eval_interval=20,
                      early_stopping=100,
                      use_val=True,
                      save_path=MODEL_PATH)


    model.load_state_dict(torch.load(MODEL_PATH))
    device = torch.device(DEVICE)
    model = model.to(device)
    raw_data.to(device)


    remove_edges = [100, 200, 300, 400, 500, 600]
    edge_list = []
    for enum in remove_edges:
        updated_edges = adjust_graph_topology_opt(raw_data,
                                                  model=model,
                                                  u_bound = u_bound,
                                                  l_bound = l_bound,
                                                  edge_to_remove=enum, use_topoinf=use_topoinf, lambda_=lambda_, default_inf=default_inf)
        data.edge_index = updated_edges
        updated_edges = updated_edges.cpu().numpy() # (2, 9938)

        edge_list.append(updated_edges.reshape(-1).tolist()) # flatten the array
        
        if val:
            edge = updated_edges[updated_edges != -1].reshape(2, -1)
            val_data = raw_data.clone()
            val_data.edge_index = torch.from_numpy(edge)
            val_acc = []
            for model_index in EVAL_MODEL_PATHs:
                result = inference_wrapper(val_data, os.path.join("models/Layer_2_Hidden_32", model_index))
                val_acc.append(round(result["val_acc"].item(), ndigits=3))
            mean = np.array(val_acc).mean().round(3)

            print(f"Result when of {edge.shape[1] // 2} edges: Mean {mean} | Original: {val_acc}")



    # NOTE: Don't change this, used for generating the submission csv
    df = pd.DataFrame(edge_list).fillna(-1).astype(int)
    # fill those empty units with -1 (don't change it)
    df.insert(0, 'ID', list(range(len(edge_list))))
    df.to_csv(save_path, index=False)
    return save_path

def from_submission(save_path):
    raw_data = torch.load(os.path.join("data","data.pt"))
    eval_model_path =  os.listdir("models/Layer_2_Hidden_32")

    df = pd.read_csv(save_path)
    edge_arr = df.to_numpy()[:, 1:]
    edge_list = []
    for i in range(edge_arr.shape[0]):
        edge_list.append(edge_arr[i][edge_arr[i] != -1].reshape(2, -1))

    means = []

    for edge in edge_list:
        raw_data.edge_index = torch.from_numpy(edge)

        val_acc = []
        for model_id in eval_model_path:
            result = inference_wrapper(raw_data, os.path.join("models/Layer_2_Hidden_32", model_id))
            val_acc.append(round(result["val_acc"].item(), ndigits=3))
        mean = np.array(val_acc).mean().round(3)
        means.append(mean)
        print(f"Result when of {edge.shape[1] // 2} edges: Mean {mean} | Original: {val_acc}")

    print("Overall mean:", np.array(means).mean().round(4))


if __name__ == "__main__":
    save_path = from_adaedge(l_bound=0.3, u_bound=1, val=True, use_topoinf=True, lambda_= 0)
    