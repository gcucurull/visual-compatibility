"""
Generic Dataloader with common functions.
"""

import numpy as np

class Dataloader(object):
    def __init__(self, path):
        """
        Args:
            path: path of the dataset
            normalize: normalize the features or not
        """
        self.path_dataset = path

        print('initializing dataloader...')
        self.init_phase('train')
        self.init_phase('valid')
        self.init_phase('test')

    def normalize_features(self, feats, get_moments=False, mean=None, std=None):
        reuse_mean = mean is not None and std is not None
        if feats.shape[1] != 2048: # image features
            raise ValueError('Features are expected to be 2048-dimensional (extracted with ResNet)')
        else:
            feats = np.array(feats.todense())
            if reuse_mean:
                mean_feats = mean
                std_feats = std
            else:
                mean_feats = feats.mean(axis=0)
                std_feats = feats.std(axis=0)

            # normalize
            feats = (feats - mean_feats)/std_feats
        if get_moments:
            return feats, mean_feats, std_feats
        return feats

    def get_negative_training_edge(self, poss_nodes, num_nodes, lower_adj):
        u = np.random.randint(num_nodes)
        v = np.random.randint(num_nodes)

        # while lower_adj[u_sample, v_sample] == 1 or lower_adj[v_sample, u_sample] == 1: # sampled a positive edge
        while lower_adj[u, v] or lower_adj[v, u]: # sampled a positive edge
            u = np.random.randint(num_nodes)

        # assert lower_adj[u_sample, v_sample] == 0
        # assert u_sample < v_sample; assert u < v;  assert u != v

        return u,v
