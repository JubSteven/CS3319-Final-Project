'''
Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-05-31 15:33:13
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-06-09 00:52:02
FilePath: /GNN/CS3319-Final-Project/run.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pickle
import os
import torch
from utils import sample_graph_community
from baseline import inference_wrapper, train_wrapper, GCN_Net
from dataset import *
from train_adaedge import adjust_graph_topology, adjust_graph_topology_opt
from train_topoinf_easy import adjust_graph_topology_topoinf_easy

DEVICE = "mps"  # cuda


LAYER_NUM = 2
HIDDEN_DIM = 32

MODEL_PATH = "/Users/husky/Desktop/GNN/CS3319-Final-Project/ada_model.pt"

# LAYER_NUM = 4
# HIDDEN_DIM = 256
# MODEL_PATH = f"models/Layer_{LAYER_NUM}_Hidden_{HIDDEN_DIM}/model_1.pt"



def from_graph(tau=0.01):
    raw_data = torch.load("data/data.pt")
    data_loader = GraphData("data/data.pt")
    eval_model_path =  os.listdir("models/Layer_2_Hidden_32")

    graph_aug = "graph_1_logits.pkl"  # NOTE: You can change this to a different graph

    with open(os.path.join("graphs", graph_aug), "rb") as f:
        adj = pickle.load(f)

    # No AUG - baseline (no delete edges)
    # [0.792, 0.796, 0.808, 0.796, 0.794, 0.800]

    remove_edges = [100, 200, 300, 400, 500, 600]
    edge_list = []
    means = []
    for enum in remove_edges:
        edges = sample_graph_community(data_loader.adj_train, adj, enum, tau)
        raw_data.edge_index = torch.from_numpy(edges.T)

        val_acc = []
        for model_id in eval_model_path:
            result = inference_wrapper(raw_data, os.path.join("models/Layer_2_Hidden_32", model_id))
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
    df.to_csv(f'submission_{tau}.csv', index=False)

    mean = np.array(means).mean().round(4)

    return mean


def from_submission():
    raw_data = torch.load("data/data.pt")
    eval_model_path =  os.listdir("models/Layer_2_Hidden_32")

    df = pd.read_csv('submission.csv')
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


def from_adaedge():
    torch.manual_seed(42)

    raw_data = torch.load('data/data.pt')
    data = torch.load('data/data.pt')
    eval_model_path =  os.listdir("models/Layer_2_Hidden_32")

    model = GCN_Net(LAYER_NUM, raw_data.num_features, HIDDEN_DIM, 7, 0.4)
    device = torch.device(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
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
        updated_edges = adjust_graph_topology_opt(raw_data,
                                                  model_path='ada_model.pt',
                                                  threshold=0.15,
                                                  edge_to_remove=enum)
        data.edge_index = updated_edges

        val_acc = []
        for model_id in eval_model_path:
            result = inference_wrapper(data, os.path.join("models/Layer_2_Hidden_32", model_id))
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

<<<<<<< HEAD

def from_topoinf_easy(lambda_=0.1):
=======
def from_topoinf_easy(lambda_ = 0.1, default_inf = -1, sim_metric = "cos", save_path = "new_submission_topoinfeasy_lambda{}_layer_{}_hidden_{}_defaultinf={}_sim={}.csv"):
>>>>>>> 0dd33c9032a62342c566f43c31c2a3e6121c836b
    torch.manual_seed(42)
    device = torch.device(DEVICE)
    save_path = save_path.format(lambda_, LAYER_NUM, HIDDEN_DIM, default_inf, sim_metric)
    
    raw_data = torch.load('data/data.pt')
    raw_data.to(device)
    model = GCN_Net(LAYER_NUM, raw_data.num_features, HIDDEN_DIM, 7, 0.4)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)


    data = torch.load('data/data.pt')
    eval_model_path =  os.listdir("models/Layer_2_Hidden_32")

    device = torch.device(DEVICE)


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
<<<<<<< HEAD
        updated_edges = adjust_graph_topology_topoinf_easy(raw_data,
                                                           model_path='ada_model.pt',
                                                           edge_to_remove=enum,
                                                           lambda_=lambda_)
=======
        updated_edges = adjust_graph_topology_topoinf_easy(raw_data, model = model, edge_to_remove= enum, lambda_ = lambda_, default_inf=default_inf, sim_metric = sim_metric)
>>>>>>> 0dd33c9032a62342c566f43c31c2a3e6121c836b
        data.edge_index = updated_edges

        val_acc = []
        for model_id in eval_model_path:
            result = inference_wrapper(data, os.path.join("models/Layer_2_Hidden_32", model_id))
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
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    for tau in range(5, 30, 5):
        mean = from_graph(tau / 1000)

    # from_submission()
    # from_adaedge()
<<<<<<< HEAD
    # from_topoinf_easy(0.5) # NOTE: You may adjust the hyperparameter lambda here
=======
    from_topoinf_easy(lambda_= 1e-7, default_inf=-1, sim_metric = "cos") # NOTE: You may adjust the hyperparameter lambda here.
>>>>>>> 0dd33c9032a62342c566f43c31c2a3e6121c836b
