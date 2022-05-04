"""
This script loads a trained model and tests it for the FITB task.
"""

import json
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import argparse
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb
import skimage.io
from collections import namedtuple
import os

from utils import get_degree_supports, sparse_to_tuple, normalize_nonsym_adj
from utils import construct_feed_dict
from model.CompatibilityGAE import CompatibilityGAE
from dataloaders import DataLoaderPolyvore, DataLoaderFashionGen

def test_fitb(args):
    args = namedtuple("Args", args.keys())(*args.values())
    load_from = args.load_from
    config_file = load_from + '/results.json'
    log_file = load_from + '/log.json'

    with open(config_file) as f:
        config = json.load(f)
    with open(log_file) as f:
        log = json.load(f)
    
    with open('data/polyvore/dataset/id2idx_test.json') as f:
        id2idx_test = json.load(f)
    
    with open('data/polyvore/dataset/id2their_test.json') as f:
        id2their_test = json.load(f)

    DATASET = config['dataset']
    NUMCLASSES = 2
    BN_AS_TRAIN = False
    ADJ_SELF_CONNECTIONS = True

    def norm_adj(adj_to_norm):
        return normalize_nonsym_adj(adj_to_norm)

    dl = DataLoaderPolyvore()
    train_features, adj_train, train_labels, train_r_indices, train_c_indices = dl.get_phase('train')
    val_features, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase('valid')
    test_features, adj_test, test_labels, test_r_indices, test_c_indices = dl.get_phase('test')
    adj_q, q_r_indices, q_c_indices, q_labels, q_ids, q_valid = dl.get_test_questions()
    train_features, mean, std = dl.normalize_features(train_features, get_moments=True)
    val_features = dl.normalize_features(val_features, mean=mean, std=std)
    test_features = dl.normalize_features(test_features, mean=mean, std=std)

    train_support = get_degree_supports(adj_train, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    val_support = get_degree_supports(adj_val, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    test_support = get_degree_supports(adj_test, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    q_support = get_degree_supports(adj_q, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)

    for i in range(1, len(train_support)):
        train_support[i] = norm_adj(train_support[i])
        val_support[i] = norm_adj(val_support[i])
        test_support[i] = norm_adj(test_support[i])
        q_support[i] = norm_adj(q_support[i])

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

    test_feed_dict = construct_feed_dict(placeholders, test_features, test_support,
                        test_labels, test_r_indices, test_c_indices, 0., is_train=BN_AS_TRAIN)
    q_feed_dict = construct_feed_dict(placeholders, test_features, q_support,
                        q_labels, q_r_indices, q_c_indices, 0., is_train=BN_AS_TRAIN)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sigmoid = lambda x: 1/(1+np.exp(-x))

    def get_image_id(id):
        K_1 = None
        for k,v in id2idx_test.items():
            if v == id:
                K_1 = k
                break

        return id2their_test[K_1]
    
    def save_image(id):
        outfit_id, index = id.split('_') # outfitID_index
        images_path = 'polyvore_images/images/'
        image_path = images_path + outfit_id + '/' + '{}.jpg'.format(index)
        assert os.path.exists(image_path)
        im = skimage.io.imread(image_path)
        if len(im.shape) == 2:
            im = gray2rgb(im)
        if im.shape[2] == 4:
            im = rgba2rgb(im)
        im = resize(im, (256, 256))
        skimage.io.imsave(f"{id}.png", im)


    with tf.Session() as sess:
        saver.restore(sess, load_from+'/'+'best_epoch.ckpt')

        kwargs = {'K': args.k, 'subset': args.subset,
                'resampled': args.resampled, 'expand_outfit':args.expand_outfit}

        for question_adj, out_ids, choices_ids, labels, valid in dl.yield_test_questions_K_edges(**kwargs):
            q_support = get_degree_supports(question_adj, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS, verbose=False)
            for i in range(1, len(q_support)):
                q_support[i] = norm_adj(q_support[i])
            q_support = [sparse_to_tuple(sup) for sup in q_support]

            q_feed_dict = construct_feed_dict(placeholders, test_features, q_support,
                            q_labels, out_ids, choices_ids, 0., is_train=BN_AS_TRAIN)

            # compute the output (correct or not) for the current FITB question
            preds = sess.run(model.outputs, feed_dict=q_feed_dict)
            preds = sigmoid(preds)
            outs = preds.reshape((-1, 4))
            outs = outs.mean(axis=0) # pick the item with average largest probability, averaged accross all edges

            gt = labels.reshape((-1, 4)).mean(axis=0)
            predicted = outs.argmax()
            gt = gt.argmax()

            for v in np.unique(out_ids):
                id = get_image_id(v)
                save_image(id)
            
            for v in  np.unique(choices_ids):
                id = get_image_id(v)
                save_image(id)
            
            print(get_image_id(choices_ids[gt]))

            break

if __name__ == "__main__":
    # TODO: remove unnecessary arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=1,
                    help="K used for the variable number of edges case")
    parser.add_argument('-eo', '--expand_outfit', dest='expand_outfit', action='store_true',
                        help='Expand the outfit nodes as well, rather than using them by default')
    parser.add_argument('-resampled', '--resampled', dest='resampled', action='store_true',
                        help='Runs the test with the resampled FITB tasks (harder)')
    parser.add_argument('-subset', '--subset', dest='subset', action='store_true',
                        help='Use only a subset of the nodes that form the outfit (3 of them) and use the others as connections')
    parser.add_argument("-lf", "--load_from", type=str, required=True, default=None, help="Model used.")
    args = parser.parse_args()
    test_fitb(vars(args))
