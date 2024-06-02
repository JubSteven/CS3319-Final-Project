'''
Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-05-31 15:33:13
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-06-02 14:38:07
FilePath: /GNN/CS3319-Final-Project/run.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pickle
import os
import torch
from utils import sample_graph_community
from baseline import inference_wrapper, train_wrapper, GCN_Net
from dataset import *
from train_adaedge import adjust_graph_topology
from train_topoinf import adjust_graph_topology_topoinf

DEVICE = "mps" # cuda

def from_graph():
    raw_data = torch.load("data/data.pt")
    data_loader = GraphData("data/data.pt")
    eval_model_path = os.listdir("models")

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
        for model in eval_model_path:
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
    raw_data = torch.load("data/data.pt")
    eval_model_path = os.listdir("models")

    df = pd.read_csv('submission-6.csv')
    edge_arr = df.to_numpy()[:, 1:]
    edge_list = []
    for i in range(edge_arr.shape[0]):
        edge_list.append(edge_arr[i][edge_arr[i] != -1].reshape(2, -1))

    means = []

    for edge in edge_list:
        raw_data.edge_index = torch.from_numpy(edge)

        val_acc = []
        for model in eval_model_path:
            result = inference_wrapper(raw_data, os.path.join("models", model))
            val_acc.append(round(result["val_acc"].item(), ndigits=3))
        mean = np.array(val_acc).mean().round(3)
        means.append(mean)
        print(f"Result when of {edge.shape[1] // 2} edges: Mean {mean} | Original: {val_acc}")

    print("Overall mean:", np.array(means).mean().round(4))


def from_adaedge():
    torch.manual_seed(42)

    raw_data = torch.load('data/data.pt')
    data = torch.load('data/data.pt')
    eval_model_path = os.listdir("models")

    model = GCN_Net(2, raw_data.num_features, 32, 7, 0.4)
    device = torch.device(DEVICE)
    model.to(device)
    raw_data.to(device)

    if not os.path.exists('ada_model.pt'):
        train_wrapper(raw_data,
                      model,
                      max_epoches=800,
                      lr=0.0002,
                      weight_decay=0.0005,
                      eval_interval=20,
                      early_stopping=100,
                      use_val=True,
                      save_path=f'ada_model.pt')

    remove_edges = [100, 200, 300, 400, 500, 600]
    edge_list = []
    means = []
    for enum in remove_edges:
        updated_edges = adjust_graph_topology(raw_data, model_path='ada_model.pt',threshold=0.15, edge_to_remove=enum) #adjust lambda_ as a hyperparameter
        data.edge_index = updated_edges

        val_acc = []
        for model in eval_model_path:
            result = inference_wrapper(data, os.path.join("models", model))
            val_acc.append(round(result["val_acc"].item(), ndigits=3))
        mean = np.array(val_acc).mean().round(3)
        print(f"Result when {enum} edges removed: Mean {mean} | Original: {val_acc}")

        means.append(mean)
        updated_edges = updated_edges.cpu().numpy()
        edge_list.append(updated_edges.reshape(-1).tolist())

    print("Overall mean:", np.array(means).mean().round(4))

    # NOTE: Don't change this, used for generating the submission csv
    df = pd.DataFrame(edge_list).fillna(-1).astype(int)
    # fill those empty units with -1 (don't change it)
    df.insert(0, 'ID', list(range(len(edge_list))))
    df.to_csv('submission.csv', index=False)
    
    

def from_topoinf(lamda = 0.1):
    torch.manual_seed(42)
    from train_topoinf import adjust_graph_topology_topoinf
    raw_data = torch.load('data/data.pt')
    data = torch.load('data/data.pt')
    eval_model_path = os.listdir("models")

    model = GCN_Net(2, raw_data.num_features, 32, 7, 0.4)
    device = torch.device(DEVICE)
    model.to(device)
    raw_data.to(device)

    remove_edges = [100, 200, 300, 400, 500, 600]
    edge_list = []
    means = []
    for enum in remove_edges:
        updated_edges = adjust_graph_topology_topoinf(raw_data, model_path='ada_model.pt',  edge_to_remove=enum, lambda_=lamda) #adjust lambda_ as a hyperparameter
        data.edge_index = updated_edges

        val_acc = []
        for model in eval_model_path:
            result = inference_wrapper(data, os.path.join("models", model))
            val_acc.append(round(result["val_acc"].item(), ndigits=3))
        mean = np.array(val_acc).mean().round(3)
        print(f"Result when {enum} edges removed: Mean {mean} | Original: {val_acc}")

        means.append(mean)
        updated_edges = updated_edges.cpu().numpy()
        edge_list.append(updated_edges.reshape(-1).tolist())

    print("Overall mean:", np.array(means).mean().round(4))

    # NOTE: Don't change this, used for generating the submission csv
    df = pd.DataFrame(edge_list).fillna(-1).astype(int)
    # fill those empty units with -1 (don't change it)
    df.insert(0, 'ID', list(range(len(edge_list))))
    df.to_csv("submission_lmd={}.csv".format(lamda), index=False)



if __name__ == "__main__":
    # from_graph()
    # from_submission()
    from_adaedge()
    # from_topoinf(0.3)
