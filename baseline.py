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
        x = F.dropout(x, p=self.dropout, training=self.training)    # NOTE: there is a dropout layer.
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
    
def train(data, model, opt):
    model.train()
    opt.zero_grad()
    x, label, edge_index, train_mask = data.x, data.y, data.edge_index, data.train_mask
    pred = model(x, edge_index)
    loss = model.loss(label, train_mask)
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
        f"Val Acc:{eval_result['val_acc']*100:6.2f} | "
        )
    
def train_warpper(data, model, max_epoches, lr=0.001, weight_decay=5e-4, eval_interval=10, early_stopping=100, early_stopping_tolerance=1):
    model.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    val_acc_history = []
    start_time = time.time()
    for epoch in tqdm(range(1, max_epoches+1)):
        loss = train(data, model, opt)
        if epoch % eval_interval == 0:
            eval_result = val(data, model)
            print_eval_result(eval_result, prefix=f'[Epoch {epoch:3d}/{max_epoches:3d}]')
            if eval_result['val_acc'] > best_val_acc:
                best_val_acc = eval_result['val_acc']
                best_model_param = copy.deepcopy(model.state_dict())
            val_acc_history.append(eval_result['val_acc'])
            if early_stopping > 0 and len(val_acc_history) > early_stopping:
                    mean_val_acc = torch.tensor(
                        val_acc_history[-(early_stopping + 1):-1]).mean().item()
                    if (eval_result['val_acc'] - mean_val_acc) * 100 < - early_stopping_tolerance: # NOTE: in percentage
                        print('[Early Stop Info] Stop at Epoch: ', epoch)
                        break
    train_time = time.time() - start_time
    model.load_state_dict(best_model_param)
    # Dump the best model
    torch.save(model.state_dict(), 'model.pt')
    eval_result = val(data, model)
    print_eval_result(eval_result, prefix=f'[Final Result] Time: {train_time:.2f}s |')



def inference_wrapper(data, model_path='model.pt'):
    """
        Input:
            data: torch_geometric.data.Data
            model_path: str
    """
    model = GCN_Net(2, data.num_features, 32, 7, 0.4)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    data.to(device)
    results = val(data, model)
    return results

if __name__ == "__main__":
    data = torch.load('data\data.pt')
    model = GCN_Net(2, data.num_features, 32, 7, 0.4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    data.to(device)
    train_warpper(data, model, max_epoches=800, lr=0.0002, weight_decay=0.0005, eval_interval=20, early_stopping=100)