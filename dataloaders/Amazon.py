import numpy as np
import scipy.sparse as sp
import time
import os

# TODO: clean unused code

class DataLoaderAmazon(object):
    """
    Load amazon data.
    """
    def __init__(self, cat_rel='Men_bought_together'):
        """
        Args:
            normalize: normalize the features or not
            cat_rel: category and type of relation used
        """
        super(DataLoaderAmazon, self).__init__()
        self.cat_rel = cat_rel

        self.path_dataset = 'data/amazon/dataset/'+cat_rel+'/'
        assert os.path.exists(self.path_dataset)

        print('initializing dataloader...')
        self.init_dataset()

    def init_dataset(self):
        path_dataset = self.path_dataset
        adj_file = path_dataset + 'adj.npz'
        feats_file = path_dataset + 'feats.npy'
        np.random.seed(1234)

        self.adj = sp.load_npz(adj_file).astype(np.int32)
        node_features = np.load(feats_file)
        self.features = node_features

        # get lower tiangle of the adj matrix to avoid duplicate edges
        self.lower_adj = sp.tril(self.adj).tocsr()

        # get positive edges and split them into train, val and test
        pos_r_idx, pos_c_idx = self.lower_adj.nonzero()
        pos_labels = np.array(self.lower_adj[pos_r_idx, pos_c_idx]).squeeze()

        n_pos = pos_labels.shape[0] # number of positive edges
        perm = list(range(n_pos))
        np.random.shuffle(perm)
        pos_labels, pos_r_idx, pos_c_idx = pos_labels[perm], pos_r_idx[perm], pos_c_idx[perm]
        n_train = int(n_pos*0.65)
        n_val = int(n_pos*0.17)

        self.train_pos_labels, self.train_pos_r_idx, self.train_pos_c_idx = pos_labels[:n_train], pos_r_idx[:n_train], pos_c_idx[:n_train]
        self.val_pos_labels, self.val_pos_r_idx, self.val_pos_c_idx = pos_labels[n_train:n_train + n_val], pos_r_idx[n_train:n_train + n_val], pos_c_idx[n_train:n_train + n_val]
        self.test_pos_labels, self.test_pos_r_idx, self.test_pos_c_idx = pos_labels[n_train + n_val:], pos_r_idx[n_train + n_val:], pos_c_idx[n_train + n_val:]

    def get_phase(self, phase):
        print('get phase: {}'.format(phase))
        assert phase in ['train', 'valid', 'test']

        lower_adj = self.lower_adj

        # get the positive edges

        if phase == 'train':
            pos_labels, pos_r_idx, pos_c_idx = self.train_pos_labels, self.train_pos_r_idx, self.train_pos_c_idx
        elif phase == 'valid':
            pos_labels, pos_r_idx, pos_c_idx = self.val_pos_labels, self.val_pos_r_idx, self.val_pos_c_idx
        elif phase == 'test':
            pos_labels, pos_r_idx, pos_c_idx = self.test_pos_labels, self.test_pos_r_idx, self.test_pos_c_idx

        # build adj matrix
        full_adj = sp.csr_matrix((
                    np.hstack([pos_labels, pos_labels]),
                    (np.hstack([pos_r_idx, pos_c_idx]), np.hstack([pos_c_idx, pos_r_idx]))
                ),
                shape=(lower_adj.shape[0], lower_adj.shape[0])
            )
        setattr(self, 'full_{}_adj'.format(phase), full_adj)

        # split the positive edges into the ones used for evaluation and the ones used as message passing
        n_pos = pos_labels.shape[0] # number of positive edges
        n_eval = int(n_pos/2)
        mp_pos_labels, mp_pos_r_idx, mp_pos_c_idx = pos_labels[n_eval:], pos_r_idx[n_eval:], pos_c_idx[n_eval:]
        # this are the positive examples that will be used to compute the loss function
        eval_pos_labels, eval_pos_r_idx, eval_pos_c_idx = pos_labels[:n_eval], pos_r_idx[:n_eval], pos_c_idx[:n_eval]

        # get the negative edges

        print('Sampling negative edges...')
        before = time.time()
        n_train_neg = eval_pos_labels.shape[0] # set the number of negative training edges that will be needed to sample at each iter
        neg_labels = np.zeros((n_train_neg))
        # get the possible indexes to be sampled (basically all indexes if there aren't restrictions)
        poss_nodes = np.arange(lower_adj.shape[0])

        neg_r_idx = np.zeros((n_train_neg))
        neg_c_idx = np.zeros((n_train_neg))

        for i in range(n_train_neg):
            r_idx, c_idx = self.get_negative_training_edge(poss_nodes, poss_nodes.shape[0], lower_adj)
            neg_r_idx[i] = r_idx
            neg_c_idx[i] = c_idx
        print('Sampling done, time elapsed: {}'.format(time.time() - before))

        # build adj matrix
        adj = sp.csr_matrix((
                    np.hstack([mp_pos_labels, mp_pos_labels]),
                    (np.hstack([mp_pos_r_idx, mp_pos_c_idx]), np.hstack([mp_pos_c_idx, mp_pos_r_idx]))
                ),
                shape=(lower_adj.shape[0], lower_adj.shape[0])
            )
        # remove the labels of the negative edges which are 0
        adj.eliminate_zeros()

        labels = np.append(eval_pos_labels, neg_labels)
        r_idx = np.append(eval_pos_r_idx, neg_r_idx)
        c_idx = np.append(eval_pos_c_idx, neg_c_idx)

        return self.features, adj, labels, r_idx, c_idx

    def normalize_features(self, feats, get_moments=False, mean=None, std=None):
        reuse_mean = mean is not None and std is not None
        if feats.shape[1] == 4096: # image features
            if reuse_mean:
                mean_feats = mean
                std_feats = std
            else:
                mean_feats = feats.mean(axis=0)
                std_feats = feats.std(axis=0)

            # normalize
            feats = (feats - mean_feats)/std_feats

        else:
            raise NotImplementedError()

        if get_moments:
            return feats, mean_feats, std_feats
        return feats

    def get_negative_training_edge(self, poss_nodes, num_nodes, lower_adj):
        """
        Sample negative training edges.
        """
        keep_search = True
        while keep_search: # sampled a positive edge
            v = np.random.randint(num_nodes)
            u = np.random.randint(num_nodes)

            keep_search = lower_adj[v, u] == 1 or lower_adj[u, v] == 1

        # assert lower_adj[v_sample, s_sample] == 0
        # assert u_sample < v_sample; assert u < v;  assert u != v

        return u,v
