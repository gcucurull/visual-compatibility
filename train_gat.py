import argparse
import time
from model import MultiGAT
import networkx as nx

import tensorflow as tf

import numpy as np
import scipy.sparse as sp
import json
import os
import shutil

from dataloaders import DataLoaderPolyvore

# Set random seed
seed = int(time.time()) # 12342
np.random.seed(seed)
tf.random.set_seed(
    seed
)


# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="polyvore",
                choices=['fashiongen', 'polyvore', 'amazon'],
                help="Dataset string.")

ap.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                help="Learning rate")

ap.add_argument("-wd", "--weight_decay", type=float, default=0.,
                help="Learning rate")

ap.add_argument("-e", "--epochs", type=int, default=4000,
                help="Number training epochs")

ap.add_argument("-hi", "--hidden", type=int, nargs='+', default=[350, 350, 350],
                help="Number hidden units in the GCN layers.")

ap.add_argument("-do", "--dropout", type=float, default=0.5,
                help="Dropout fraction")

ap.add_argument("-deg", "--degree", type=int, default=1,
                help="Degree of the convolution (Number of supports)")

ap.add_argument("-sdir", "--summaries_dir", type=str, default="logs/",
                help="Directory for saving tensorflow summaries.")

ap.add_argument("-sup_do", "--support_dropout", type=float, default=0.15,
                help="Use dropout on the support matrices, dropping all the connections from some nodes")

ap.add_argument('-ws', '--write_summary', dest='write_summary', default=False,
                help="Option to turn on summary writing", action='store_true')

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-bn', '--batch_norm', dest='batch_norm',
                help="Option to turn on batchnorm in GCN layers", action='store_true')
fp.add_argument('-no_bn', '--no_batch_norm', dest='batch_norm',
                help="Option to turn off batchnorm", action='store_false')
ap.set_defaults(batch_norm=True)

ap.add_argument("-amzd", "--amz_data", type=str, default="Men_bought_together",
            choices=['Men_also_bought', 'Women_also_bought', 'Women_bought_together', 'Men_bought_together'],
            help="Dataset string.")

args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')

# Define parameters
DATASET = args['dataset']
NB_EPOCH = args['epochs']
DO = args['dropout']
HIDDEN = args['hidden']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = "img"
NUMCLASSES = 2
DEGREE = args['degree']
BATCH_NORM = args['batch_norm']
BN_AS_TRAIN = False
SUP_DO = args['support_dropout']
ADJ_SELF_CONNECTIONS = True
VERBOSE = True

# prepare data_loader
dl = DataLoaderPolyvore()
train_features, adj_train, train_labels, train_r_indices, train_c_indices = dl.get_phase('train')
val_features, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase('valid')
G = nx.from_scipy_sparse_matrix(adj_train, create_using=nx.Graph)
edges = np.array(G.edges)
G_val = nx.from_scipy_sparse_matrix(adj_val, create_using=nx.Graph)
edges_val = np.array(G_val.edges)
train_features = train_features.toarray()
val_features = val_features.toarray()

train_r_indices = train_r_indices.astype('int')
train_c_indices = train_c_indices.astype('int')
val_r_indices = val_r_indices.astype('int')
val_c_indices = val_c_indices.astype('int')

mul_attn = MultiGAT.GAEAttn()
train_labels = train_labels.reshape(train_labels.shape[0], 1)
val_labels = val_labels.reshape(val_labels.shape[0], 1)

val_best = [0, 0]
checkpoint = tf.train.Checkpoint(mul_attn)
manager = tf.train.CheckpointManager(checkpoint, directory="tmp/model", max_to_keep=5)
status = checkpoint.restore(manager.latest_checkpoint)
for i in range(NB_EPOCH):
    loss, acc = mul_attn.train_step(train_labels, [train_features, edges], train_r_indices, train_c_indices)
    print(f'Epoch {i+1} : {loss} , {acc}') 
    loss, acc = mul_attn.train_step(val_labels, [val_features, edges_val], val_r_indices, val_c_indices)
    if acc > val_best[1]:
        manager.save()
        val_best = [loss, acc]
        print(val_best)

print(val_best)