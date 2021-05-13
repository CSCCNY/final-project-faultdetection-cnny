import sys
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCNConv, GraphSageConv, ChebConv
from spektral.models.gcn import GCN
from spektral.transforms import AdjToSpTensor, LayerPreprocess, NormalizeOne
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from circuit_dataset import CircuitDataset

x_data = []

# the path should have files with name {circuit_name}.x, {circuit_name}.y and {circuit_name}.graph
def load_data(circuit_name, path_to_data="data", normalize=False):
    """Load data."""
    names = ["x", "y", "graph"]
    objects = []
    for i in range(len(names)):
        with open(f"{path_to_data}/{circuit_name}.{names[i]}", "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, graph = tuple(objects)
    x = np.array(x).astype("float32")
    for x_feat in x:
        x_data.append(x_feat)
    if normalize:
        x = normalize_features(x)

    features = sp.csr_matrix(x).astype("float32")

    g = nx.DiGraph()
    g.add_nodes_from(graph.keys())

    # add self loops
    for k, v in graph.items():
        g.add_edges_from(([(k, t) for t in v]))
        g.add_edges_from([(k, k)])
    adj = nx.adjacency_matrix(g)
    labels = np.array(y).astype("float32").reshape((-1, 1))

    return adj, features, labels


def row_level_features_preprocessing(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_features(x):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x.toarray())


def normalize_feature(circ_dataset):
    scaler = MinMaxScaler()
    scaler.fit(x_data)
    for graph in circ_dataset:
        graph.x = scaler.transform(graph.x)
    return circ_dataset


def train_test_val_data(dataset, epochs=400, batch_size=1, path="../data/"):
    # dataset = CircuitDataset(path=path, transforms=transforms)
    # # Parameters
    # F = dataset.n_node_features  # Dimension of node features
    # n_out = dataset.n_labels  # Dimension of the target

    # Train/valid/test split
    idxs = np.random.permutation(len(dataset))
    split_va, split_te = int(0.6 * len(dataset)), int(0.8 * len(dataset))
    idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
    print(idx_tr, idx_va, idx_te)

    dataset_tr = dataset[idx_tr]
    dataset_va = dataset[idx_va]
    dataset_te = dataset[idx_te]

    loader_tr = DisjointLoader(
        dataset_tr, batch_size=batch_size, epochs=epochs, node_level=True
    )
    loader_va = DisjointLoader(dataset_va, batch_size=batch_size, node_level=True)
    loader_te = DisjointLoader(dataset_te, batch_size=batch_size, node_level=True)
    return loader_tr, loader_va, loader_te


def load_path(path):
    sys.path.append(path)


def remove_path(path):
    sys.path.remove(path)
