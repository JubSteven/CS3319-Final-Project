import pickle
import os
import torch
from utils import sample_graph_det, to_submission
from baseline import inference_wrapper
from dataset import *

raw_data = torch.load("data\data.pt")
data_loader = GraphData("data\data.pt")

graph_aug = "graph_3_logits.pkl"  # You can change this to a different graph

with open(os.path.join("graphs", graph_aug), "rb") as f:
    adj = pickle.load(f)

remove_edges = [100, 200, 300, 400, 500, 600]
edge_list = []
for enum in remove_edges:
    edges = sample_graph_det(data_loader.adj_train.to_dense().numpy(), adj, enum)
    raw_data.edge_index = torch.from_numpy(edges.T)
    result = inference_wrapper(raw_data, "model.pt")
    print(f"Result when {enum} edges removed: {result}, current edge count: {raw_data.edge_index.shape[1] // 2}")

    edge_list.append(edges.T.reshape(-1).tolist())

df = pd.DataFrame(edge_list).fillna(-1).astype(int)
# fill those empty units with -1 (don't change it)
df.insert(0, 'ID', list(range(len(edge_list))))
df.to_csv('submission.csv', index=False)
