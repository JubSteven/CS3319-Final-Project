import torch
from utils import *


class GraphData():

    def __init__(self, data_path, device="cpu", use_louvain=True, louvain_neighbors=10, louvain_lambda=0.5):
        self.data = torch.load(data_path)
        self.device = device
        self.use_louvain = use_louvain
        self.louvain_neighbors = louvain_neighbors
        self.louvain_lambda = louvain_lambda
        self.init_data()

    def init_data(self):
        self.x, self.y, self.edge_index, self.val_mask, self.train_mask = self.data.x, self.data.y, self.data.edge_index, self.data.val_mask, self.data.train_mask

        # Filter out the validation and training data
        self.x_train = self.x[self.train_mask]
        self.edge_index_train = torch.stack(
            [edge for edge in self.edge_index.permute(1, 0) if self.train_mask[edge[0]] and self.train_mask[edge[1]]])
        self.edge_index_val = torch.stack(
            [edge for edge in self.edge_index.permute(1, 0) if self.val_mask[edge[0]] and self.val_mask[edge[1]]])

        # Adjacency matrix contains all the edges in the graph
        # NOTE: train_edges here can be all the edges or the training edges only
        self.train_edges = self.edge_index
        # ! self.x ought to be modified if train_edges is not all edges
        self.adj_train = self.build_adj(self.train_edges, self.x.shape[0]).numpy()

        if self.use_louvain:
            self.louvain_adj, _, _ = louvain_clustering(self.adj_train, self.louvain_neighbors)
            self.adj_train_louvain = self.adj_train + self.louvain_adj * self.louvain_lambda

        self.adj_train_norm = self.normalize_adj(self.adj_train)
        self.adj_train_louvain_norm = self.normalize_adj(self.adj_train_louvain)

        # ! self.x ought to be modified if train_edges is not all edges
        self.adj_label = self.build_adj_label(self.train_edges, self.x.shape[0])

        self.val_edges = self.edge_index_val
        self.val_edges_false = self.generate_false_edges(self.val_edges.shape[0])

    def generate_false_edges(self, num_edges):
        """
            Generates num_edges false edges for the graph
        """
        false_edges = set()

        while len(false_edges) < num_edges:
            src_node = np.random.randint(0, self.x.shape[0])
            dst_node = np.random.randint(0, self.x.shape[0])
            if src_node != dst_node and not self.adj_train_norm[src_node, dst_node]:
                false_edges.add((src_node, dst_node))

        false_edges = torch.tensor(list(false_edges)).to(self.device)
        return false_edges

    def sparse_to_tuple(self, sparse_mx):
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        values = sparse_mx.data
        shape = sparse_mx.shape
        return coords, values, shape

    def build_adj(self, edge_idx, num_verts, half=False):
        """
            Input:
                edge_idx: [torch.Tensor] a tensor of shape (2, num_edges) representing the edge indices of the graph
                num_verts: the number of nodes in the graph
            Returns:
                adjacency_matrix: an adjacency matrix built from edge_idx 
        """
        adjacency_matrix = torch.zeros((num_verts, num_verts))

        # Iterate over each edge and set the corresponding entries in the adjacency matrix
        for i in range(edge_idx.size(1)):
            src_node = edge_idx[0, i]
            dst_node = edge_idx[1, i]
            adjacency_matrix[src_node, dst_node] = 1
            if not half:
                adjacency_matrix[dst_node, src_node] = 1

        return adjacency_matrix

    def build_adj_label(self, edge_idx, num_verts, half=False):
        """
            Input:
                edge_idx: [torch.Tensor] a tensor of shape (2, num_edges) representing the edge indices of the graph
                num_verts: the number of nodes in the graph
            Returns:
                adj_label: [torch.sparse.FloatTensor] a sparse matrix representing the adjacency matrix label of the graph (half)
        """
        adj = self.build_adj(edge_idx, num_verts, half)
        adj_label = adj + torch.eye(num_verts)
        adj_label = adj_label.to_sparse()

        return adj_label

    def normalize_adj(self, adj_train):
        """
            Input:
                adj_train: [np.ndarray] a sparse matrix representing the adjacency matrix of the graph
            Returns:
                adj_norm: [torch.sparse.FloatTensor] a normalized version of the adjacency matrix
        """
        adj_ = sp.coo_matrix(adj_train)
        adj_.setdiag(1)
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        adj_norm_tuple = self.sparse_to_tuple(adj_norm)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_tuple[0].T), torch.FloatTensor(adj_norm_tuple[1]),
                                            torch.Size(adj_norm_tuple[2]))

        return adj_norm
