
import argparse
import time

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import scipy.sparse as sp
import json
import os
import shutil

from utils import sparse_to_tuple, get_degree_supports, normalize_nonsym_adj
from model.CompatibilityGAE import CompatibilityGAE
from utils import construct_feed_dict, write_log, support_dropout
from dataloaders import DataLoaderPolyvore, DataLoaderFashionGen, DataLoaderAmazon

# Set random seed
seed = int(time.time()) # 12342
np.random.seed(seed)
tf.set_random_seed(seed)

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
if DATASET in ['fashiongen', 'polyvore']:
    if DATASET == 'fashiongen':
        dl = DataLoaderFashionGen()
    elif DATASET == 'polyvore':
        dl = DataLoaderPolyvore()
    train_features, adj_train, train_labels, train_r_indices, train_c_indices = dl.get_phase('train')
    val_features, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase('valid')
    test_features, adj_test, test_labels, test_r_indices, test_c_indices = dl.get_phase('test')
    adj_q, q_r_indices, q_c_indices, q_labels, q_ids, q_valid = dl.get_test_questions()
    if DATASET == 'polyvore':
        res_adj_q, res_q_r_indices, res_q_c_indices, res_q_labels, res_q_ids, res_q_valid = dl.get_test_questions(resampled=True) # resampled
    train_features, mean, std = dl.normalize_features(train_features, get_moments=True)
    val_features = dl.normalize_features(val_features, mean=mean, std=std)
    test_features = dl.normalize_features(test_features, mean=mean, std=std)
elif DATASET == 'amazon':
    cat_rel = args['amz_data']
    dl = DataLoaderAmazon(cat_rel=cat_rel)
    train_features, adj_train, train_labels, train_r_indices, train_c_indices = dl.get_phase('train')
    _, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase('valid')
    _, adj_test, test_labels, test_r_indices, test_c_indices = dl.get_phase('test')
    train_features, mean, std = dl.normalize_features(train_features, get_moments=True)
else:
    raise NotImplementedError('A data loader for dataset {} does not exist'.format(DATASET))

if not os.path.exists(SUMMARIESDIR):
    os.makedirs(SUMMARIESDIR)

if SUMMARIESDIR == 'logs/':
    SUMMARIESDIR += str(len(os.listdir(SUMMARIESDIR)))

log_file = SUMMARIESDIR + '/log.json'
log_data = {
    'val':{'loss':[], 'acc':[]},
    'train':{'loss':[], 'acc':[]},
    'questions':{
        'loss':[], 'acc':[],
        'task_acc': [], 'task_acc_cf': [], 'res_task_acc': [],
    },
}

if not os.path.exists(SUMMARIESDIR):
    os.makedirs(SUMMARIESDIR)

train_support = get_degree_supports(adj_train, DEGREE, adj_self_con=ADJ_SELF_CONNECTIONS)
val_support = get_degree_supports(adj_val, DEGREE, adj_self_con=ADJ_SELF_CONNECTIONS)
test_support = get_degree_supports(adj_test, DEGREE, adj_self_con=ADJ_SELF_CONNECTIONS)
if DATASET != 'amazon':
    q_support = get_degree_supports(adj_q, DEGREE, adj_self_con=ADJ_SELF_CONNECTIONS)
if DATASET == 'polyvore':
    res_q_support = get_degree_supports(res_adj_q, DEGREE, adj_self_con=ADJ_SELF_CONNECTIONS)

for i in range(1, len(train_support)):
    train_support[i] = normalize_nonsym_adj(train_support[i])
    val_support[i] = normalize_nonsym_adj(val_support[i])
    test_support[i] = normalize_nonsym_adj(test_support[i])
    if DATASET != 'amazon':
        q_support[i] = normalize_nonsym_adj(q_support[i])
    if DATASET == 'polyvore':
        res_q_support[i] = normalize_nonsym_adj(res_q_support[i])    

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
                    hidden=HIDDEN,
                    learning_rate=LR,
                    logging=True,
                    batch_norm=BATCH_NORM,
                    wd=args['weight_decay'])

# Feed_dicts for validation and test set stay constant over different update steps
train_feed_dict = construct_feed_dict(placeholders, train_features, train_support,
                    train_labels, train_r_indices, train_c_indices, DO)
if DATASET != 'amazon':
    val_feed_dict = construct_feed_dict(placeholders, val_features, val_support,
                        val_labels, val_r_indices, val_c_indices, 0., is_train=BN_AS_TRAIN)
    test_feed_dict = construct_feed_dict(placeholders, test_features, test_support,
                        test_labels, test_r_indices, test_c_indices, 0., is_train=BN_AS_TRAIN)
    q_feed_dict = construct_feed_dict(placeholders, test_features, q_support,
                        q_labels, q_r_indices, q_c_indices, 0., is_train=BN_AS_TRAIN)
else:
    val_feed_dict = construct_feed_dict(placeholders, train_features, val_support,
                        val_labels, val_r_indices, val_c_indices, 0., is_train=BN_AS_TRAIN)
    test_feed_dict = construct_feed_dict(placeholders, train_features, test_support,
                        test_labels, test_r_indices, test_c_indices, 0., is_train=BN_AS_TRAIN)

# Collect all variables to be logged into summary
merged_summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if WRITESUMMARY:
    train_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/val')
else:
    train_summary_writer = None
    val_summary_writer = None

best_val_score = 0
best_train_score = 0
best_epoch_train_score = 0
best_val_loss = np.inf
best_epoch = 0
wait = 0

print('Training...')

for epoch in range(NB_EPOCH):
    t = time.time()

    # modify train_feed_dict with support dropout if needed
    if SUP_DO:
        # do not modify the first support, the self-connections one
        for i in range(1, len(train_support)):
            modified = support_dropout(train_support[i].copy(), SUP_DO, edge_drop=True)
            modified.data[...] = 1 # make it binary to normalize
            modified = normalize_nonsym_adj(modified)
            modified = sparse_to_tuple(modified)
            train_feed_dict.update({placeholders['support'][i]: modified})

    # run one iteration
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.confmat], feed_dict=train_feed_dict)
    
    train_avg_loss = outs[1]
    train_acc = outs[2]

    val_avg_loss, val_acc, conf = sess.run([model.loss, model.accuracy, model.confmat], feed_dict=val_feed_dict)

    if VERBOSE:
        print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_loss=", "{:.5f}".format(val_avg_loss),
              "val_acc=", "{:.5f}".format(val_acc),
              "\t\ttime=", "{:.5f}".format(time.time() - t))

    log_data['train']['loss'].append(float(train_avg_loss))
    log_data['train']['acc'].append(float(train_acc))
    log_data['val']['loss'].append(float(val_avg_loss))
    log_data['val']['acc'].append(float(val_acc))

    write_log(log_data, log_file)

    if val_acc > best_val_score:
        best_val_score = val_acc
        best_epoch = epoch
        best_epoch_train_score = train_acc
        saver = tf.train.Saver()
        save_path = saver.save(sess, "%s/best_epoch.ckpt" % (SUMMARIESDIR))

    if train_acc > best_train_score:
        best_train_score = train_acc

    if epoch % 50 == 0 and WRITESUMMARY:
        # Train set summary
        summary = sess.run(merged_summary, feed_dict=train_feed_dict)
        train_summary_writer.add_summary(summary, epoch)
        train_summary_writer.flush()

        # Validation set summary
        summary = sess.run(merged_summary, feed_dict=val_feed_dict)
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()

# store model
saver = tf.train.Saver()
save_path = saver.save(sess, "%s/%s.ckpt" % (SUMMARIESDIR, model.name), global_step=model.global_step)

if VERBOSE:
    print("\nOptimization Finished!")
    print('best validation score =', best_val_score, 'at iteration {}, with a train_score of {}'.format(best_epoch, best_epoch_train_score))

print('\nSETTINGS:\n')
for key, val in sorted(vars(ap.parse_args()).items()):
    print(key, val)

print('global seed = ', seed)

# For parsing results from file
results = vars(ap.parse_args()).copy()
results.update({'best_val_score': float(best_val_score), 'best_epoch': best_epoch})
results.update({'best_epoch_train_score': float(best_epoch_train_score)})
results.update({'best_train_score': float(best_train_score)})
results.update({'best_epoch': best_epoch})
results.update({'seed':seed})

print(json.dumps(results))

json_outfile = SUMMARIESDIR + '/' + 'results.json'
with open(json_outfile, 'w') as outfile:
    json.dump(results, outfile)

sess.close()
