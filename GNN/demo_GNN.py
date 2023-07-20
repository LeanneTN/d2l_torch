"""
Implementing a GNN in PyTorch from scratch without using PyG
"""
import networkx as nx
import torch
from torch.nn.parameter import Parameter
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class NodeNetwork(torch.nn.Module):
    """
    1. perform two graph convolutions (self.conv1 and self.conv2)
    2. pool all the node embeddings via global_sum_pool
    3. run the pooled embeddings through two fully connected layers
    4. output a class-membership probability via softmax
    The structure of this network could be viewed in the Machine Learning with PyTorch and Scikit-learn P650
    """

    def __init__(self, input_features):
        super().__init__()
        self.conv1 = BasicGraphConvolutionLayer(input_features, 32)
        self.conv2 = BasicGraphConvolutionLayer(32, 32)
        self.fc1 = torch.nn.Linear(32, 16)
        self.out_layer = torch.nn.Linear(16, 2)

    def forward(self, X, A, batch_mat):
        x = F.relu(self.conv1(X, A))
        x = F.relu(self.conv2(x, A))
        output = global_sum_pool(x, batch_mat)
        output = self.fc1(output)
        output = self.out_layer(output)
        return F.softmax(output, dim=1)


class BasicGraphConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W2 = Parameter(torch.rand((in_channels, out_channels), dtype=torch.float32))
        self.W1 = Parameter(torch.rand((in_channels, out_channels), dtype=torch.float32))
        self.bias = Parameter(torch.zeros(out_channels, dtype=torch.float32))

    def forward(self, X, A):
        """
        With graph convolutions, we can interpret each row of X as being an embedding of the information that is
        stored at the node corresponding to that row. Graph convolutions update the embeddings at each node based on
        the embeddings of their neighbors and themselves. This procedure can be written as:
        $x_i' = x_iW_1 + \sum_{j \in N(i)} x_jW_2 + b$
        :param X: Input vector
        :param A: Adjacent Matrix
        :return:
        """
        potential_msgs = torch.mm(X, self.W2)
        propagated_msgs = torch.mm(A, potential_msgs)
        root_update = torch.mm(X, self.W1)
        output = propagated_msgs + root_update + self.bias
        return output


def global_sum_pool(X, batch_mat):
    """Adding a global pooling layer to deal with varying graph sizes"""
    """
    Why using batch_mat:
    Graph sizes vary and approach is not feasible with graph data unless padding is used. The way to deal with it is to
    treat each batch as a single graph where each graph in the batch is a subgraph that is disconnected from the rest
    Batch_mat is to serve as a graph selection mask that keeps the graphs in the batch separate.
    """
    if batch_mat is None or batch_mat.dim() == 1:
        return torch.sum(X, dim=0).unsqueeze(0)
    else:
        return torch.mm(batch_mat, X)


def get_batch_tensor(graph_size):
    """
    To generate the batch mask used in the global pooling
    :param graph_size: N x N
    :return: batch_mat[i-1, si:si + ni] = 1
    """
    starts = [sum(graph_size[:idx]) for idx in range(len(graph_size))]
    stops = [starts[idx] + graph_size[idx] for idx in range(len(graph_size))]
    tot_len = sum(graph_size)
    batch_size = len(graph_size)
    batch_mat = torch.zeros([batch_size, tot_len]).float()
    for idx, starts_and_stops in enumerate(zip(starts, stops)):
        start = starts_and_stops[0]
        stop = starts_and_stops[1]
        batch_mat[idx, start:stop] = 1
    return batch_mat


def collate_graphs(batch):
    """batch is a list of dictionaries each containing the representation and label of a graph"""
    adj_mats = [graph['A'] for graph in batch]
    sizes = [A.size(0) for A in adj_mats]
    tot_size = sum(sizes)
    # create batch matrix
    batch_mat = get_batch_tensor(sizes)
    # combine feature matrices
    feat_mats = torch.cat([graph['X'] for graph in batch], dim=0)
    # combine labels
    labels = torch.cat([graph['y'] for graph in batch], dim=0)
    # combine adjacency matrices
    batch_adj = torch.zeros([tot_size, tot_size], dtype=torch.float32)
    accum = 0
    for adj in adj_mats:
        g_size = adj.shape[0]
        batch_adj[accum:accum + g_size, accum:accum + g_size] = adj
        accum = accum + g_size
    repr_and_label = {'A': batch_adj, 'X': feat_mats, 'y': labels, 'batch': batch_mat}
    return repr_and_label


def build_graph_color_label_representation(G, mapping_dict):
    one_hot_idxs = np.array([mapping_dict[v] for v in nx.get_node_attributes(G, 'color').values()])
    one_hot_encoding = np.zeros((one_hot_idxs.size, len(mapping_dict)))
    one_hot_encoding[np.arange(one_hot_idxs.size), one_hot_idxs] = 1
    return one_hot_encoding


def get_graph_dict(G, mapping_dict):
    # Function builds dictionary representation of graph G
    A = torch.from_numpy(np.asarray(nx.adjacency_matrix(G).todense())).float()
    X = torch.from_numpy(build_graph_color_label_representation(G, mapping_dict)).float()
    y = torch.tensor([[1, 0]]).float()
    return {'A': A, 'X': X, 'y': y, 'batch': None}


class ExampleDataset(Dataset):
    def __init__(self, graph_list):
        self.graphs = graph_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        mol_rep = self.graphs[idx]
        return mol_rep


if __name__ == '__main__':
    blue, orange, green = '#1f77b4', '#ff7f0e', '#2ca02c'
    mapping_dict = {green: 0, blue: 1, orange: 2}
    G1 = nx.Graph()
    G1.add_nodes_from([
        (1, {"color": blue}),
        (2, {"color": orange}),
        (3, {"color": blue}),
        (4, {"color": green})
    ])
    G1.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4)])
    G2 = nx.Graph()
    G2.add_nodes_from([
        (1, {"color": green}),
        (2, {"color": green}),
        (3, {"color": orange}),
        (4, {"color": orange}),
        (5, {"color": blue})
    ])
    G2.add_edges_from([(2, 3), (3, 4), (3, 1), (5, 1)])
    G3 = nx.Graph()
    G3.add_nodes_from([
        (1, {"color": orange}),
        (2, {"color": orange}),
        (3, {"color": green}),
        (4, {"color": green}),
        (5, {"color": blue}),
        (6, {"color": orange})
    ])
    G3.add_edges_from([(2, 3), (3, 4), (3, 1), (5, 1), (2, 5), (6, 1)])
    G4 = nx.Graph()
    G4.add_nodes_from([
        (1, {"color": blue}),
        (2, {"color": blue}),
        (3, {"color": green})
    ])
    G4.add_edges_from([(1, 2), (2, 3)])
    graph_list = [get_graph_dict(graph, mapping_dict) for graph in [G1, G2, G3, G4]]

    dset = ExampleDataset(graph_list)
    loader = DataLoader(dset, batch_size=2, shuffle=False, collate_fn=collate_graphs)

    # prediction (this simple GNN has no learning process (backward propagation)
    node_feature = 3
    net = NodeNetwork(node_feature)
    batch_results = []
    for b in loader:
        result = net(b['X'], b['A'], b['batch']).detach()
        batch_results.append(result)
        print(result)

    G1_rep = dset[1]
    G1_single = net(G1_rep['X'], G1_rep['A'], G1_rep['batch']).detach()
    G1_batch = batch_results[0][1]
    print(torch.all(torch.isclose(G1_single, G1_batch)))
