"""
This script computes the compatibility score of each outfit, by processing each
outfit as an independent graph.
"""

import json
import tensorflow as tf
import argparse
import numpy as np
import scipy.sparse as sp
import time
from sklearn.metrics import roc_auc_score
from collections import namedtuple

from utils import get_degree_supports, sparse_to_tuple, normalize_nonsym_adj
from utils import construct_feed_dict, Graph
from model.CompatibilityGAE import CompatibilityGAE
from dataloaders import DataLoaderPolyvore, DataLoaderFashionGen

def compute_auc(preds, labels):
    return roc_auc_score(labels.astype(int), preds)

def test_compatibility(args):
    args = namedtuple("Args", args.keys())(*args.values())
    load_from = args.load_from
    config_file = load_from + '/results.json'
    log_file = load_from + '/log.json'

    with open(config_file) as f:
        config = json.load(f)
    with open(log_file) as f:
        log = json.load(f)

    # Dataloader
    DATASET = config['dataset']
    if DATASET == 'polyvore':
        # load dataset
        dl = DataLoaderPolyvore()
        orig_train_features, adj_train, train_labels, train_r_indices, train_c_indices = dl.get_phase('train')
        full_train_adj = dl.train_adj
        orig_val_features, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase('valid')
        orig_test_features, adj_test, test_labels, test_r_indices, test_c_indices = dl.get_phase('test')
        full_test_adj = dl.test_adj
        dl.setup_test_compatibility(resampled=args.resampled)
    elif DATASET == 'ssense':
        dl = DataLoaderFashionGen()
        orig_train_features, adj_train, train_labels, train_r_indices, train_c_indices = dl.get_phase('train')
        orig_val_features, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase('valid')
        orig_test_features, adj_test, test_labels, test_r_indices, test_c_indices = dl.get_phase('test')
        adj_q, q_r_indices, q_c_indices, q_labels, q_ids, q_valid = dl.get_test_questions()
        full_train_adj = dl.train_adj
        full_test_adj = dl.test_adj
        dl.setup_test_compatibility(resampled=args.resampled)
    else:
        raise NotImplementedError('A data loader for dataset {} does not exist'.format(DATASET))

    NUMCLASSES = 2
    BN_AS_TRAIN = False
    ADJ_SELF_CONNECTIONS = True

    def norm_adj(adj_to_norm):
        return normalize_nonsym_adj(adj_to_norm)

    train_features, mean, std = dl.normalize_features(orig_train_features, get_moments=True)
    val_features = dl.normalize_features(orig_val_features, mean=mean, std=std)
    test_features = dl.normalize_features(orig_test_features, mean=mean, std=std)

    train_support = get_degree_supports(adj_train, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    val_support = get_degree_supports(adj_val, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    test_support = get_degree_supports(adj_test, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)

    for i in range(1, len(train_support)):
        train_support[i] = norm_adj(train_support[i])
        val_support[i] = norm_adj(val_support[i])
        test_support[i] = norm_adj(test_support[i])

    num_support = len(train_support)
    placeholders = {
        'row_indices': tf.placeholder(tf.int32, shape=(None,)),
        'col_indices': tf.placeholder(tf.int32, shape=(None,)),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'weight_decay': tf.placeholder_with_default(0., shape=()),
        'is_train': tf.placeholder_with_default(True, shape=()),
        'support': [tf.sparse_placeholder(tf.float32, shape=(None, None)) for sup in range(num_support)],
        'node_features': tf.placeholder(tf.float32, shape=(None, None)),
        'labels': tf.placeholder(tf.float32, shape=(None,))   
    }

    model = CompatibilityGAE(placeholders,
                    input_dim=train_features.shape[1],
                    num_classes=NUMCLASSES,
                    num_support=num_support,
                    hidden=config['hidden'],
                    learning_rate=config['learning_rate'],
                    logging=True,
                    batch_norm=config['batch_norm'])

    # Construct feed dicts for train, val and test phases
    train_feed_dict = construct_feed_dict(placeholders, train_features, train_support,
                    train_labels, train_r_indices, train_c_indices, config['dropout'])
    val_feed_dict = construct_feed_dict(placeholders, val_features, val_support,
                        val_labels, val_r_indices, val_c_indices, 0., is_train=BN_AS_TRAIN)
    test_feed_dict = construct_feed_dict(placeholders, test_features, test_support,
                        test_labels, test_r_indices, test_c_indices, 0., is_train=BN_AS_TRAIN)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    def eval():
        # use this as a control value, if the model is ok, the value will be the same as in log
        val_avg_loss, val_acc, conf, pred = sess.run([model.loss, model.accuracy, model.confmat, model.predict()], feed_dict=val_feed_dict)
        
        print("val_loss=", "{:.5f}".format(val_avg_loss),
              "val_acc=", "{:.5f}".format(val_acc))

    with tf.Session() as sess:
        saver.restore(sess, load_from+'/'+'best_epoch.ckpt')

        count = 0
        preds = []
        labels = []

        # evaluate the the model for accuracy prediction
        eval()

        prob_act = tf.nn.sigmoid

        K = args.k
        for outfit in dl.comp_outfits:
            before_item = time.time()
            items, score = outfit

            num_new = test_features.shape[0]

            new_adj = sp.csr_matrix((num_new, num_new)) # no connections

            if args.k > 0:
                # add edges to the adj matrix
                available_adj = dl.test_adj.copy()
                available_adj = available_adj.tolil()

                i = 0
                for idx_from in items[:-1]:
                    for idx_to in items[i+1:]:
                        # remove outfit edges, they won't be expanded
                        available_adj[idx_to, idx_from] = 0
                        available_adj[idx_from, idx_to] = 0
                    i += 1
                available_adj = available_adj.tocsr()
                available_adj.eliminate_zeros()

            if args.subset: # use only a subset (of size 3) of the outfit
                items = np.random.choice(items, 3)

            new_features = test_features

            # predict edges between the items
            query_r = []
            query_c = []

            i = 0
            item_indexes = items
            for idx_from in item_indexes[:-1]:
                for idx_to in item_indexes[i+1:]:
                    query_r.append(idx_from)
                    query_c.append(idx_to)
                i += 1

            if args.k > 0:
                G = Graph(available_adj)
                nodes_to_expand = np.unique(items)
                for node in nodes_to_expand:
                    edges = G.run_K_BFS(node, K)
                    for edge in edges:
                        u, v = edge
                        new_adj[u, v] = 1
                        new_adj[v, u] = 1

            query_r = np.array(query_r)
            query_c = np.array(query_c)

            new_adj = new_adj.tocsr()

            new_support = get_degree_supports(new_adj, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS, verbose=False)
            for i in range(1, len(new_support)):
                new_support[i] = norm_adj(new_support[i])
            new_support = [sparse_to_tuple(sup) for sup in new_support]

            new_feed_dict = construct_feed_dict(placeholders, new_features, new_support,
                                        train_labels, query_r, query_c, 0., is_train=BN_AS_TRAIN)

            pred = sess.run(prob_act(model.outputs), feed_dict=new_feed_dict)

            predicted_score = pred.mean()
            print("[{}] Mean scores between outfit: {:.4f}, label: {}".format(count, predicted_score, score))
            # TODO: remove this print
            print("Total Elapsed: {:.4f}".format(time.time() - before_item))
            count += 1

            preds.append(predicted_score)
            labels.append(score)

        preds = np.array(preds)
        labels = np.array(labels)

        AUC = compute_auc(preds, labels)

        # use this as a control value, if the model is ok, the value will be the same as in log
        eval()

        print('The AUC compat score is: {}'.format(AUC))

    print('Best val score saved in log: {}'.format(config['best_val_score']))
    print('Last val score saved in log: {}'.format(log['val']['acc'][-1]))

    print("mean positive prediction: {}".format(preds[labels.astype(bool)].mean()))
    print("mean negative prediction: {}".format(preds[np.logical_not(labels.astype(bool))].mean()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lf", "--load_from", type=str,
                    default=None, help="Model used.")
    parser.add_argument('-subset', '--subset', dest='subset', action='store_true',
                        help='Use only a subset of the nodes that form the outfit (3 of them) and use the others as connections')
    parser.add_argument('-resampled', '--resampled', dest='resampled', action='store_true',
                        help='Use the resampled test, where the invalid outfits are harder.')
    parser.add_argument("-k", type=int, default=1,
                    help="K used for the variable number of edges case")
    args = parser.parse_args()
    test_compatibility(vars(args))
