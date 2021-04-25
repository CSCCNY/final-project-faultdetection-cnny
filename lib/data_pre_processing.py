import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import pdb


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(circuit_name, path_to_data="data"):
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

    features = sp.csr_matrix(x)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = y

    print(adj.shape)
    print(features.shape)
    return adj, features, labels


def preprocess_features(features):
    """Row-normalize feature matrix and convert it to dense representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def load_path(path):
    sys.path.append(path)


def remove_path(path):
    sys.path.remove(path)


# load_data("c6288", "output/")
