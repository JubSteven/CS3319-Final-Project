import pickle
import os
import torch
from utils import sample_graph_det, sample_graph_community
from baseline import inference_wrapper
from dataset import *


def from_graph():
    raw_data = torch.load("data\data.pt")
    data_loader = GraphData("data\data.pt")
    model_path = os.listdir("models")

    graph_aug = "graph_1_logits.pkl"  # NOTE: You can change this to a different graph

    with open(os.path.join("graphs", graph_aug), "rb") as f:
        adj = pickle.load(f)

    # No AUG - baseline (no delete edges)
    # [0.792, 0.796, 0.808, 0.796, 0.794, 0.800]

    remove_edges = [100, 200, 300, 400, 500, 600]
    edge_list = []
    means = []
    for enum in remove_edges:
        edges = sample_graph_community(data_loader.adj_train, adj, enum, tau=0.01)
        raw_data.edge_index = torch.from_numpy(edges.T)

        val_acc = []
        for model in model_path:
            result = inference_wrapper(raw_data, os.path.join("models", model))
            val_acc.append(round(result["val_acc"].item(), ndigits=3))
        mean = np.array(val_acc).mean().round(3)
        print(f"Result when {enum} edges removed: Mean {mean} | Original: {val_acc}")

        means.append(mean)
        edge_list.append(edges.T.reshape(-1).tolist())

    print("Overall mean:", np.array(means).mean().round(4))

    # NOTE: Don't change this, used for generating the submission csv
    df = pd.DataFrame(edge_list).fillna(-1).astype(int)
    # fill those empty units with -1 (don't change it)
    df.insert(0, 'ID', list(range(len(edge_list))))
    df.to_csv('submission.csv', index=False)


def from_submission():
    raw_data = torch.load("data\data.pt")
    model_path = os.listdir("models")

    df = pd.read_csv('submission.csv')
    edge_arr = df.to_numpy()[:, 1:]
    edge_list = []
    for i in range(edge_arr.shape[0]):
        edge_list.append(edge_arr[i][edge_arr[i] != -1].reshape(2, -1))

    means = []

    for edge in edge_list:
        raw_data.edge_index = torch.from_numpy(edge)

        val_acc = []
        for model in model_path:
            result = inference_wrapper(raw_data, os.path.join("models", model))
            val_acc.append(round(result["val_acc"].item(), ndigits=3))
        mean = np.array(val_acc).mean().round(3)
        means.append(mean)
        print(f"Result when of {edge.shape[1] // 2} edges: Mean {mean} | Original: {val_acc}")

    print("Overall mean:", np.array(means).mean().round(4))


if __name__ == "__main__":
    from_graph()
    # from_submission()
