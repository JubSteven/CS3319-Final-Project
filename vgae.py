import torch_geometric as pyg
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np


class VGAE(nn.Module):
    """
        The GVAE is adapted from https://github.com/zhao-tong/GAug/blob/master/vgae/models.py
    """

    def __init__(self, adj, dim_in, dim_h, dim_z, use_gae=False, adj_louvain=None):
        super(VGAE, self).__init__()

        self.dim_z = dim_z
        self.gae = use_gae
        if adj_louvain is not None:
            self.base_gcn = GraphConvSparse(dim_in, dim_h, adj_louvain)
        else:
            self.base_gcn = GraphConvSparse(dim_in, dim_h, adj)

        self.gcn_mean = GraphConvSparse(dim_h, dim_z, adj, activation=False)
        self.gcn_logstd = GraphConvSparse(dim_h, dim_z, adj, activation=False)

    def encode(self, X):
        # TODO: modify X to fuse basic graph features
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        if self.gae:
            # graph auto-encoder
            return self.mean
        else:
            # variational graph auto-encoder
            self.logstd = self.gcn_logstd(hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
            return sampled_z

    def decode(self, Z):
        A_pred = Z @ Z.T
        return A_pred

    def forward(self, X, F=None):
        if F is not None:
            # TODO design different fusing strategies
            if isinstance(F, np.ndarray):
                F = torch.from_numpy(F).float().to(X.device)
            X = torch.cat([X, F], dim=1)
        Z = self.encode(X)
        A_pred = self.decode(Z)
        return A_pred


class GraphConvSparse(nn.Module):

    def __init__(self, input_dim, output_dim, adj, activation=True):
        super(GraphConvSparse, self).__init__()
        self.weight = self.glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        return nn.Parameter(initial)

    def forward(self, inputs):
        x = inputs @ self.weight
        x = self.adj @ x
        if self.activation:
            return F.elu(x)
        else:
            return x
