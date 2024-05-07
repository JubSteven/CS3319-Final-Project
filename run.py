import pickle
import os
import torch
from utils import sample_graph_det, to_submission
from baseline import inference_wrapper
from dataset import *

raw_data = torch.load("data\data.pt")
graph_files = os.listdir("graphs")
data_loader = GraphData("data\data.pt")

for file in graph_files:
    if "logits" not in file:
        continue

    with open(os.path.join("graphs", file), "rb") as f:
        adj = pickle.load(f)

    edges = sample_graph_det(data_loader.adj_train.to_dense().numpy(), adj)
    raw_data.edge_index = torch.from_numpy(edges.T)
    result = inference_wrapper(raw_data, "model.pt")
    print(result)
