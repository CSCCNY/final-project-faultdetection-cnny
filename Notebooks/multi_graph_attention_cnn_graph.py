from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
from keras.regularizers import l2
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from utils import *
from keras_dgl.layers import MultiGraphAttentionCNN

# prepare data
################################################################################
# LOAD DATA
################################################################################
import sys
from spektral.data import Dataset, Graph
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
x_data = []
y_data = []
batch_size = 3  # Batch size
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
        
    features = sp.csr_matrix(x).astype('float32')
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).astype(int)
    g = nx.DiGraph()
    g.add_nodes_from(graph.keys())
    for k, v in graph.items():
      g.add_edges_from(([(k, t) for t in v]))
      g.add_edges_from([(k, k)])
    adj = nx.adjacency_matrix(g)
    labels = np.array(y).astype('float32').reshape((-1,1))
    for l in y:
        y_data.append(l)
    print(adj.shape)
    print(features.shape)
    return adj, features, labels
  
def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels


def _preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features



class CircuitDataset(Dataset):
    def read(self):
        circuits = []
#         circs = ['c6288','c5315','c432', 'c499', 'c17', 'c880', 'c1355', 'c1908', 'c3540', 'adder.bench', 'arbiter.bench', 'cavlc.bench', 'dec.bench', 'voter.bench', "sin.bench","priority.bench", "multiplier.bench", "max.bench"]
        circs = ["adder.bench","arbiter.bench","c1355","c1908","c3540","c432","c499","c5315","c6288","c880","cavlc.bench","dec.bench","int2float.bench","max.bench","multiplier.bench","priority.bench","sin.bench","voter.bench"]
#         circs = ["log2.bench"]
#         circs = ['adder.bench', "arbiter.bench",  "sin.bench", "multiplier.bench", "voter.bench", "priority.bench"]
        for circ in circs:
            A, X, labels = load_data(circ, '../data/output', normalize="")
#             if sum(labels) >= 500:
            print(f"{circ}: {sum(labels)}, {len(labels)}")
            circuits.append(Graph(x=X.toarray(), a=A, y=labels))

        return circuits

def normalize_feature(circ_dataset):
    scaler = MinMaxScaler()
    scaler.fit(x_data)
    for graph in circ_dataset:
        graph.x = scaler.transform(graph.x)
    return circ_dataset

dataset = normalize_feature(CircuitDataset(transforms=[AdjToSpTensor()]))

# Parameters

F = dataset.n_node_features  # Dimension of node features
n_out = dataset.n_labels  # Dimension of the target

# Train/valid/test split
np.random.seed(42)
idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
print(idx_tr, idx_va, idx_te)
dataset_tr = dataset[idx_tr]
dataset_va = dataset[idx_va]
dataset_te = dataset[idx_te]
from sklearn.utils.class_weight import compute_class_weight

y_data = []
for data_tr in dataset_tr:
    y_data.append(data_tr.y)
y_data = np.vstack((y_data)).reshape((-1,))


def _compute_class_weight_dictionary(y):
    # helper for returning a dictionary instead of an array
    classes = np.unique(y).astype('float32')
    class_weight = compute_class_weight("balanced", classes, y)
    class_weight_dict = dict(zip(classes, class_weight))
    return class_weight_dict 

weights = _compute_class_weight_dictionary(np.vstack((y_data)).reshape((-1,)))
y_data = []
for data_tr in dataset:
    y_data.append(data_tr.y)
y_data = np.vstack((y_data)).reshape((-1,))

scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

A, X, Y = shuffle(A, X, Y)

# build graph_conv_filters
SYM_NORM = True
num_filters = 2
graph_conv_filters = preprocess_adj_tensor_with_identity(A, SYM_NORM)

# set daigonal values to 1 in adjacency matrices
A_eye_tensor = []
for _ in range(num_graphs):
    Identity_matrix = np.eye(num_graph_nodes)
    A_eye_tensor.append(Identity_matrix)

A_eye_tensor = np.array(A_eye_tensor)
A = np.add(A, A_eye_tensor)

# build model
X_input = Input(shape=(X.shape[1], X.shape[2]))
A_input = Input(shape=(A.shape[1], A.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))

output = MultiGraphAttentionCNN(100, num_filters=num_filters, num_attention_heads=2, attention_combine='concat', attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([X_input, A_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = MultiGraphAttentionCNN(100, num_filters=num_filters, num_attention_heads=1, attention_combine='average', attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([output, A_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = Lambda(lambda x: K.mean(x, axis=1))(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
output = Dense(Y.shape[1], activation='elu')(output)
output = Activation('softmax')(output)

nb_epochs = 500
batch_size = 169

model = Model(inputs=[X_input, A_input, graph_conv_filters_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit([X, A, graph_conv_filters], Y, batch_size=batch_size, validation_split=0.1, epochs=nb_epochs, shuffle=True, verbose=1)

# sample output
# 169/169 [==============================] - 0s 138us/step - loss: 0.5014 - acc: 0.7633 - val_loss: 0.3162 - val_acc: 0.8436
# Epoch 496/500
# 169/169 [==============================] - 0s 127us/step - loss: 0.4770 - acc: 0.7633 - val_loss: 0.3187 - val_acc: 0.8436
# Epoch 497/500
# 169/169 [==============================] - 0s 131us/step - loss: 0.4781 - acc: 0.7574 - val_loss: 0.3196 - val_acc: 0.8436
# Epoch 498/500
# 169/169 [==============================] - 0s 120us/step - loss: 0.4925 - acc: 0.7574 - val_loss: 0.3197 - val_acc: 0.8436
# Epoch 499/500
# 169/169 [==============================] - 0s 137us/step - loss: 0.4911 - acc: 0.7692 - val_loss: 0.3161 - val_acc: 0.8436
# Epoch 500/500
# 169/169 [==============================] - 0s 127us/step - loss: 0.5004 - acc: 0.7633 - val_loss: 0.3130 - val_acc: 0.8436
