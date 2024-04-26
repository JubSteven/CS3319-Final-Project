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


def train_model(cfg, graph_data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # weights for log_lik loss
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
        A_pred = model(features)
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
            # print("test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
            #     r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))

        loss.backward()
        optimizer.step()
        train_bar.set_description(
            f"E: {epoch+1} | L: {loss.item():.4f} | A: {r['acc']:.4f} | ROC: {r['roc']:.4f} | AP: {r['ap']:.4f} | F1: {r['f1']:.4f}"
        )

    print("Training completed. Final results: test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".
          format(r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))

    model.load_state_dict(best_state_dict)
    # Dump the best model
    torch.save(model.state_dict(), f'models/model.pth')
    return model


def gen_graphs(cfg, graph_data, model):
    adj_orig = graph_data.adj_train

    if cfg["use_gae"]:
        pickle.dump(adj_orig, open(f'graphs/graph_0_gae.pkl', 'wb'))
    else:
        pickle.dump(adj_orig, open(f'graphs/graph_0.pkl', 'wb'))

    features = graph_data.x.to(cfg["device"])
    for i in range(cfg["gen_graphs"]):
        with torch.no_grad():
            A_pred = model(features)

        A_pred = torch.sigmoid(A_pred).detach().cpu()
        adj_recon = A_pred.numpy()
        np.fill_diagonal(adj_recon, 0)

        if cfg["use_gae"]:
            filename = f'graphs/graph_{i+1}_logits_gae.pkl'
        else:
            filename = f'graphs/graph_{i+1}_logits.pkl'

        pickle.dump(adj_recon, open(filename, 'wb'))


def update_edge(data, adj_matrix):
    edge_idx_update = adj_matrix.indices
    data.edge_index = edge_idx_update
    return data


def to_submission(path="data\data_augmented.pt"):
    data = torch.load(path)
    edge_index = data.edge_index
    edge_index = edge_index.numpy().reshape(1, -1)
    df = pd.DataFrame(edge_index)
    df.insert(0, 'ID', [0])
    df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    to_submission()
