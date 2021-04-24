import sys, inspect
import os
import joblib
import numpy as np
import h5py
import scipy.sparse.linalg as la
import scipy.sparse as sp
import scipy
import time
import pickle
import scipy.io as sio
import pdb
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.disable_v2_behavior()


class GCN:
    """
    The neural network model.
    """

    def __init__(
        self, A, X, Y, num_hidden_feat, learning_rate=5e-2, gamma=1e-3, idx_gpu="/gpu:2"
    ):

        self.num_hidden_feat = num_hidden_feat
        self.learning_rate = learning_rate
        self.gamma = gamma
        with tf.Graph().as_default() as g:
            self.graph = g

            with tf.device(idx_gpu):
                # definition of constant matrices
                self.A = convert_coo_to_sparse_tensor(A.tocoo())
                self.X = tf.constant(X, dtype=tf.float32)
                self.Y = tf.constant(Y, dtype=tf.float32)

                self.W0 = tf.get_variable(
                    "W0",
                    shape=[X.shape[1], self.num_hidden_feat],
                    initializer=tf.keras.layers.xavier_initializer(),
                )
                self.W1 = tf.get_variable(
                    "W1",
                    shape=[self.num_hidden_feat, Y.shape[1]],
                    initializer=tf.keras.layers.xavier_initializer(),
                )

                # placeholder definition
                self.idx_nodes = tf.placeholder(tf.int32)
                self.keep_prob = tf.placeholder(tf.float32)

                # model definition
                self.l_input = tf.nn.dropout(self.X, self.keep_prob)

                self.X0_tilde = tf.sparse_tensor_dense_matmul(self.A, self.l_input)
                self.X0 = tf.matmul(self.X0_tilde, self.W0)
                self.X0 = tf.nn.relu(self.X0)
                self.X0 = tf.nn.dropout(self.X0, self.keep_prob)

                self.X1_tilde = tf.sparse_tensor_dense_matmul(self.A, self.X0)
                self.logits = tf.matmul(self.X1_tilde, self.W1)

                self.l_out = tf.gather(self.logits, self.idx_nodes)
                self.c_Y = tf.gather(self.Y, self.idx_nodes)

                # loss function definition
                self.l2_reg = tf.nn.l2_loss(self.W0)
                self.data_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.l_out, labels=self.c_Y
                    )
                )
                self.loss = self.data_loss + self.gamma * self.l2_reg

                # solver definition
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                )
                self.opt_step = self.optimizer.minimize(self.loss)

                # predictions and accuracy extraction
                self.c_predictions = tf.argmax(tf.nn.softmax(self.l_out), 1)
                self.accuracy = tf.contrib.metrics.accuracy(
                    self.c_predictions, tf.argmax(self.c_Y, 1)
                )

                # gradients computation
                self.trainable_variables = tf.trainable_variables()
                self.var_grad = tf.gradients(self.loss, tf.trainable_variables())
                self.norm_grad = frobenius_norm(
                    tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0)
                )

                # session creation
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.session = tf.Session(config=config)

                # session initialization
                init = tf.global_variables_initializer()
                self.session.run(init)


def convert_coo_to_sparse_tensor(L):
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data.astype("float32"), L.shape)
    L = tf.sparse.reorder(L)
    return L


def frobenius_norm(tensor):
    square_tensor = tf.square(tensor)
    tensor_sum = tf.reduce_sum(square_tensor)
    frobenius_norm = tf.sqrt(tensor_sum)
    return frobenius_norm
