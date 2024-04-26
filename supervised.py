import torch_geometric as pyg
import torch
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from gvae import *
from utils import *
from dataset import *


def train(data_path, cfg):

    graph_data = GraphData(data_path, cfg["device"])
    model = GVAE(graph_data.adj_matrix, graph_data.x.shape[1], cfg["dim_h"], cfg["dim_z"],
                 use_gae=cfg["use_gae"]).to(cfg["device"])
    train_model(cfg, graph_data, model)


if __name__ == "__main__":
    # dim_h represents the hidden size, while dim_z represents the embedding size
    # ENCODE[X -> h -> Z] -> DECODE[\hat{X}]
    cfg = {
        "dim_h": 32,
        "dim_z": 16,
        "lr": 0.01,
        "epoch": 200,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_gae": False,
        "criterion": "roc",
        "gen_graphs": 10,
    }
    train(data_path="data\data.pt", cfg=cfg)
