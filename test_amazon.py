import json
import time
import tensorflow as tf
import argparse
import numpy as np
import scipy.sparse as sp
from collections import namedtuple

from utils import get_degree_supports, sparse_to_tuple, normalize_nonsym_adj
from utils import construct_feed_dict, Graph
from model.CompatibilityGAE import CompatibilityGAE
from dataloaders import DataLoaderAmazon

def test_amazon(args):
    args = namedtuple("Args", args.keys())(*args.values())

    load_from = args.load_from
    config_file = load_from + '/results.json'
    log_file = load_from + '/log.json'

    with open(config_file) as f:
        config = json.load(f)
    with open(log_file) as f:
        log = json.load(f)

    NUMCLASSES = 2 
    BN_AS_TRAIN = False
    ADJ_SELF_CONNECTIONS = True

    # evaluate in the specified version
    print("Trained with {}, evaluating with {}".format(config['amz_data'], args.amz_data))
    cat_rel = args.amz_data
    dp = DataLoaderAmazon(cat_rel=cat_rel)
    train_features, adj_train, train_labels, train_r_indices, train_c_indices = dp.get_phase('train')
    _, adj_val, val_labels, val_r_indices, val_c_indices = dp.get_phase('valid')
    _, adj_test, test_labels, test_r_indices, test_c_indices = dp.get_phase('test')
    full_adj = dp.adj

    def norm_adj(adj_to_norm):
        return normalize_nonsym_adj(adj_to_norm)

    train_features, mean, std = dp.normalize_features(train_features, get_moments=True)

    train_support = get_degree_supports(adj_train, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    val_support = get_degree_supports(adj_val, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    test_support = get_degree_supports(adj_test, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)

    for i in range(1, len(train_support)):
        train_support[i] = norm_adj(train_support[i])
        val_support[i] = norm_adj(val_support[i])
        test_support[i] = norm_adj(test_support[i])

    num_support = len(train_support)

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

    train_feed_dict = construct_feed_dict(placeholders, train_features, train_support,
                                          train_labels, train_r_indices, train_c_indices, config['dropout'])
    # No dropout for validation and test runs
    val_feed_dict = construct_feed_dict(placeholders, train_features, val_support,
                                        val_labels, val_r_indices, val_c_indices, 0., is_train=BN_AS_TRAIN)
    test_feed_dict = construct_feed_dict(placeholders, train_features, test_support,
                                         test_labels, test_r_indices, test_c_indices, 0., is_train=BN_AS_TRAIN)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, load_from+'/'+'best_epoch.ckpt')

        val_avg_loss, val_acc, conf, pred = sess.run([model.loss, model.accuracy, model.confmat, model.predict()], feed_dict=val_feed_dict)

        print("val_loss=", "{:.5f}".format(val_avg_loss),
              "val_acc=", "{:.5f}".format(val_acc))

        test_avg_loss, test_acc, conf = sess.run([model.loss, model.accuracy, model.confmat], feed_dict=test_feed_dict)

        print("test_loss=", "{:.5f}".format(test_avg_loss),
              "test_acc=", "{:.5f}".format(test_acc))

        # rerun for K=0 (all in parallel)
        k_0_adj = sp.csr_matrix(adj_val.shape)
        k_0_support = get_degree_supports(k_0_adj, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS, verbose=False)
        for i in range(1, len(k_0_support)):
            k_0_support[i] = norm_adj(k_0_support[i])
        k_0_support = [sparse_to_tuple(sup) for sup in k_0_support]

        k_0_val_feed_dict = construct_feed_dict(placeholders, train_features, k_0_support,
                                            val_labels, val_r_indices, val_c_indices, 0., is_train=BN_AS_TRAIN)
        k_0_test_feed_dict = construct_feed_dict(placeholders, train_features, k_0_support,
                                            test_labels, test_r_indices, test_c_indices, 0., is_train=BN_AS_TRAIN)

        val_avg_loss, val_acc, conf, pred = sess.run([model.loss, model.accuracy, model.confmat, model.predict()], feed_dict=k_0_val_feed_dict)
        print("for k=0 val_loss=", "{:.5f}".format(val_avg_loss),
              "for k=0 val_acc=", "{:.5f}".format(val_acc))

        test_avg_loss, test_acc, conf = sess.run([model.loss, model.accuracy, model.confmat], feed_dict=k_0_test_feed_dict)
        print("for k=0 test_loss=", "{:.5f}".format(test_avg_loss),
              "for k=0 test_acc=", "{:.5f}".format(test_acc))

        K = args.k

        available_adj = dp.full_valid_adj + dp.full_train_adj
        available_adj = available_adj.tolil()
        for r,c in zip(test_r_indices, test_c_indices):
            available_adj[r,c] = 0
            available_adj[c,r] = 0
        available_adj = available_adj.tocsr()
        available_adj.eliminate_zeros()

        G = Graph(available_adj)
        get_edges_func = G.run_K_BFS

        new_adj = sp.csr_matrix(full_adj.shape)
        new_adj = new_adj.tolil()
        for r,c in zip(test_r_indices, test_c_indices):
            before = time.time()
            if K > 0: #expand the edges
                nodes_to_expand = [r,c]
                for node in nodes_to_expand:
                    edges = get_edges_func(node, K)
                    for edge in edges:
                        i, j = edge
                        new_adj[i, j] = 1
                        new_adj[j, i] = 1

        new_adj = new_adj.tocsr()

        new_support = get_degree_supports(new_adj, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS, verbose=False)
        for i in range(1, len(new_support)):
            new_support[i] = norm_adj(new_support[i])
        new_support = [sparse_to_tuple(sup) for sup in new_support]

        new_feed_dict = construct_feed_dict(placeholders, train_features, new_support,
                            test_labels, test_r_indices, test_c_indices, 0., is_train=BN_AS_TRAIN)

        loss, acc = sess.run([model.loss, model.accuracy], feed_dict=new_feed_dict)

        print("for k={} test_acc=".format(K), "{:.5f}".format(acc))

    print('Best val score saved in log: {}'.format(config['best_val_score']))
    print('Last val score saved in log: {}'.format(log['val']['acc'][-1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=0,
                    help="K used for the variable number of edges case")
    parser.add_argument("-lf", "--load_from", type=str, help="Model used.")
    parser.add_argument("-amzd", "--amz_data", type=str, default="Men_bought_together",
            choices=['Men_also_bought', 'Women_also_bought', 'Women_bought_together', 'Men_bought_together'],
            help="Dataset string.")
    args = parser.parse_args()
    test_amazon(vars(args))