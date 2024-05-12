import torch_geometric as pyg
import torch
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from vgae import *
from utils import *
from dataset import *


def train_model(cfg, graph_data, model, structural_features=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    adj_t = graph_data.adj_train
    norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
    pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(cfg["device"])

    # move input data and label to gpu if needed
    features = graph_data.x.to(cfg["device"])
    adj_label = graph_data.adj_label.to_dense().to(cfg["device"])

    best_vali_criterion = 0.0
    best_state_dict = None
    model.train()

    train_bar = tqdm(range(cfg["epoch"]))
    for epoch in train_bar:
        A_pred = model(X=features, F=structural_features)
        optimizer.zero_grad()
        loss = norm_w * F.binary_cross_entropy_with_logits(A_pred, adj_label, pos_weight=pos_weight)
        if not cfg["use_gae"]:
            kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * model.logstd - model.mean**2 -
                                                    torch.exp(2 * model.logstd)).sum(1).mean()
            loss -= kl_divergence

        A_pred = torch.sigmoid(A_pred).detach().cpu()
        r = get_scores(graph_data.val_edges, graph_data.val_edges_false, A_pred, graph_data.adj_label)

        if r[cfg["criterion"]] > best_vali_criterion:
            best_vali_criterion = r[cfg["criterion"]]
            best_state_dict = copy.deepcopy(model.state_dict())
            r_test = r

        loss.backward()
        optimizer.step()
        train_bar.set_description(
            f"E: {epoch+1} | L: {loss.item():.4f} | A: {r['acc']:.4f} | ROC: {r['roc']:.4f} | AP: {r['ap']:.4f} | F1: {r['f1']:.4f}"
        )

    print("Training completed. Final results: test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".
          format(r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))

    model.load_state_dict(best_state_dict)
    # Dump the best model
    torch.save(model.state_dict(), f'aug_model.pt')
    return model


def gen_graphs(cfg, graph_data, model, structural_features=None):
    adj_orig = graph_data.adj_train

    if cfg["use_gae"]:
        pickle.dump(adj_orig, open(f'graphs/graph_0_gae.pkl', 'wb'))
    else:
        pickle.dump(adj_orig, open(f'graphs/graph_0.pkl', 'wb'))

    features = graph_data.x.to(cfg["device"])
    for i in range(cfg["gen_graphs"]):
        with torch.no_grad():
            A_pred = model(features, structural_features)

        A_pred = torch.sigmoid(A_pred).detach().cpu()
        adj_recon = A_pred.numpy()
        np.fill_diagonal(adj_recon, 0)

        if cfg["use_gae"]:
            filename = f'graphs/graph_{i+1}_logits_gae.pkl'
        else:
            filename = f'graphs/graph_{i+1}_logits.pkl'

        pickle.dump(adj_recon, open(filename, 'wb'))


def main(data_path, cfg):
    graph_data = GraphData(data_path, cfg["device"])  # TODO: validate the correctness of the dataloader
    # structural_features = get_basic_graph_features(graph_data.adj_train)

    model = VGAE(
        adj=graph_data.adj_train_norm,
        adj_louvain=graph_data.adj_train_louvain if cfg["use_louvain"] else None,
        dim_in=graph_data.x.shape[1],
        dim_h=cfg["dim_h"],
        dim_z=cfg["dim_z"],
        use_gae=cfg["use_gae"],
    ).to(cfg["device"])

    if cfg["pretrained"]:
        model.load_state_dict(torch.load(cfg["pretrained"]))
    else:
        model = train_model(cfg, graph_data, model)

    if cfg["gen_graphs"] > 0:
        # Generate graphs
        gen_graphs(cfg, graph_data, model)


if __name__ == "__main__":
    # dim_h represents the hidden size, while dim_z represents the embedding size
    # ENCODE[X -> h -> Z] -> DECODE[\hat{X}]
    cfg = {
        "pretrained": None,
        "dim_h": 32,
        "dim_z": 16,
        "lr": 0.01,
        "epoch": 200,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_gae": True,
        "criterion": "roc",
        "gen_graphs": 1,
        "use_louvain": True,
    }
    main(data_path="data\data.pt", cfg=cfg)
