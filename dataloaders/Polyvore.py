import numpy as np
import scipy.sparse as sp
import json
import time

from .Dataloader import Dataloader

class DataLoaderPolyvore(Dataloader):
    """
    Load polyvore data.
    """
    def __init__(self):
        super(DataLoaderPolyvore, self).__init__(path='data/polyvore/dataset/')

    def init_phase(self, phase):
        print('init phase: {}'.format(phase))
        assert phase in ['train', 'valid', 'test']
        path_dataset = self.path_dataset
        adj_file = path_dataset + 'adj_{}.npz'.format(phase)
        feats_file = path_dataset + 'features_{}.npz'.format(phase)
        np.random.seed(1234)

        adj = sp.load_npz(adj_file).astype(np.int32)
        setattr(self, '{}_adj'.format(phase), adj)
        node_features = sp.load_npz(feats_file)
        setattr(self, '{}_features'.format(phase), node_features)

        # get lower tiangle of the adj matrix to avoid duplicate edges
        setattr(self, 'lower_{}_adj'.format(phase), sp.tril(adj).tocsr())

        questions_file = path_dataset + 'questions_test.json'
        questions_file_resampled = questions_file.replace('questions', 'questions_RESAMPLED')
        with open(questions_file) as f:
            self.questions = json.load(f)
        with open(questions_file_resampled) as f:
            self.questions_resampled = json.load(f)

    def get_phase(self, phase):
        print('get phase: {}'.format(phase))
        assert phase in ['train', 'valid', 'test']

        lower_adj = getattr(self, 'lower_{}_adj'.format(phase))

        # get the positive edges

        pos_r_idx, pos_c_idx = lower_adj.nonzero()
        pos_labels = np.array(lower_adj[pos_r_idx, pos_c_idx]).squeeze()

        # split the positive edges into the ones used for evaluation and the ones used as message passing
        n_pos = pos_labels.shape[0] # number of positive edges
        perm = list(range(n_pos))
        np.random.shuffle(perm)
        pos_labels, pos_r_idx, pos_c_idx = pos_labels[perm], pos_r_idx[perm], pos_c_idx[perm]
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

        return getattr(self, '{}_features'.format(phase)), adj, labels, r_idx, c_idx

    def get_test_questions(self, resampled=False):
        """
        Return the FITB 'questions' in form of node indexes
        """
        # each question consists on N*4 edges to predict
        # self.questions is a list of questions with N elements and 4 possible choices (answers)
        flat_questions = []
        gt = []
        q_ids = [] # this list indicates to which question does each edge belongs to
        valid = []
        q_id = 0
        questions = self.questions if not resampled else self.questions_resampled
        for question in questions:
            for index in question[0]: # indexes of outfit nodes
                i = 0
                for index_answer in question[1]: # indexes of possible choices answers
                    flat_questions.append([index, index_answer]) # append the edge
                    if i == 0:
                        gt.append(1) # the correct connection is the first
                    else:
                        gt.append(0)
                    # a link is valid if the candidate item is from the same category as the missing item
                    valid.append(int(question[2][i] == question[3]))
                    i += 1
                    q_ids.append(q_id)
            q_id += 1

        assert len(flat_questions) == len(gt) and len(q_ids) == len(gt) and len(gt) == len(valid)
        assert len(self.questions) == max(q_ids)+1

        # flat questions contains edges [u,v]
        # gt contains the ground truth label for each edge
        # q_ids indicates to which question does each query edge belong to

        flat_questions = np.array(flat_questions)
        gt = np.array(gt)
        q_ids = np.array(q_ids)
        valid = np.array(valid)

        # now build the adj for message passing for the questions, by removing the edges that will be evaluated
        lower_adj = getattr(self, 'lower_{}_adj'.format('test'))

        full_adj = lower_adj + lower_adj.transpose()
        full_adj = full_adj.tolil()
        for edge, label in zip(flat_questions, gt):
            u, v = edge
            full_adj[u, v] = 0
            full_adj[v, u] = 0

        full_adj = full_adj.tocsr()
        full_adj.eliminate_zeros()

        # make sure that none of the query edges are in the adj matrix
        count_edges = 0
        count_pos = 0
        for edge in flat_questions:
            u,v = edge
            if full_adj[u,v] > 0:
                count_pos += 1
            count_edges += 1
        assert count_pos == 0

        return full_adj, flat_questions[:, 0], flat_questions[:, 1], gt, q_ids, valid

    def yield_test_questions_K_edges(self, resampled=False, K=1, subset=False, expand_outfit=False):
        """
        Yields questions, each of them with their own adj matrix.
        Each node on the question will be expanded to create K edges, so the adj
        matrix will have K*N edges.
        Also, the edges between the nodes of the outfit will be also present (except for the correct choice edges).
        The method to get this edges will be BFS.

        Args:
            resampled: if True, use the resampled version
            K: number of edges to expand for each question node
            subset: if true, use only a subset of the outfit as the query, and
                    use the rest as links to the choices.

        Returns:
            yields questions
        """
        assert K >= 0
        from utils import Graph

        # each question consists on N*4 edges to predict
        # self.questions is a list of questions with N elements and 4 possible choices (answers)
        questions = self.questions if not resampled else self.questions_resampled
        n_nodes = self.test_adj.shape[0]
        for question in questions:
            outfit_ids = []
            choices_ids = []
            gt = []
            valid = []
            # keep only a subset of the outfit
            if subset:
                outfit_subset = np.random.choice(question[0], 3, replace=False)
            else:
                outfit_subset = question[0]
            for index in outfit_subset: # indexes of outfit nodes
                i = 0
                for index_answer in question[1]: # indexes of possible choices answers
                    outfit_ids.append(index)
                    choices_ids.append(index_answer)
                    gt.append(int(i==0))# the correct connection is the first
                    # a link is valid if the candidate item is from the same category as the missing item
                    valid.append(int(question[2][i] == question[3]))
                    i += 1

            # question adj with only the outfit edges
            question_adj = sp.csr_matrix((n_nodes, n_nodes))
            question_adj = question_adj.tolil()
            if not expand_outfit:
                for j,u in enumerate(outfit_subset[:-1]):
                    for v in outfit_subset[j+1:]:
                        question_adj[u, v] = 1
                        question_adj[v, u] = 1

            if K > 0:
                # the K edges that will be sampled from each not will not belong to the outfit, and should not be the query edges, so remove them
                available_adj = self.test_adj.copy()
                available_adj = available_adj.tolil()
                for j,u in enumerate(question[0][:-1]):
                    for v in question[0][j+1:]:
                        available_adj[u, v] = 0
                        available_adj[v, u] = 0
                if expand_outfit: # activate intra-outfit edges
                    for j,u in enumerate(outfit_subset[:-1]):
                        for v in outfit_subset[j+1:]:
                            available_adj[u,v] = 1
                            available_adj[v,u] = 1
                for u, v in zip(outfit_ids, choices_ids):
                    available_adj[u, v] = 0
                    available_adj[v, u] = 0
                available_adj = available_adj.tocsr()
                available_adj.eliminate_zeros()

                G = Graph(available_adj)

                extra_edges = []
                # now fill the adj matrix with the expanded edges for each node (only for the choices)
                nodes_to_expand = choices_ids[:4]

                if expand_outfit: # expand the outfit items as well
                    nodes_to_expand.extend(outfit_subset)

                for node in nodes_to_expand:
                    edges = G.run_K_BFS(node, K)

                    for edge in edges:
                        u, v = edge
                        question_adj[u, v] = 1
                        question_adj[v, u] = 1
                        extra_edges.append(edge)

            question_adj = question_adj.tocsr()

            yield question_adj, np.array(outfit_ids), np.array(choices_ids), np.array(gt), np.array(valid)

    def get_test_compatibility(self):
        """
        This function is not used now, becaue full_adj is empty because all the edges have been removed
        """
        self.setup_test_compatibility()

        flat_questions = []
        gt = []
        q_ids = []
        q_id = 0
        for outfit in self.comp_outfits:
            items = outfit[0]
            for i in range(len(items)):
                for to_idx in items[i+1:]:
                    from_idx = items[i]
                    flat_questions.append([from_idx, to_idx])
                    gt.append(outfit[1])
                    q_ids.append(q_id)
            q_id += 1

        assert len(flat_questions) == len(gt) and len(q_ids) == len(gt)
        assert len(self.comp_outfits) == max(q_ids)+1

        flat_questions = np.array(flat_questions)
        gt = np.array(gt)
        q_ids = np.array(q_ids)

        # now build the adj for message passing for the questions, by removing the edges that will be evaluated
        lower_adj = getattr(self, 'lower_{}_adj'.format('test'))

        full_adj = lower_adj + lower_adj.transpose()
        full_adj = full_adj.tolil()
        for edge, label in zip(flat_questions, gt):
            u, v = edge
            full_adj[u, v] = 0
            full_adj[v, u] = 0

        full_adj = full_adj.tocsr()
        full_adj.eliminate_zeros()

        # make sure that none of the query edges are in the adj matrix
        count_edges = 0
        count_pos = 0
        for edge in flat_questions:
            u,v = edge
            if full_adj[u,v] > 0:
                count_pos += 1
            count_edges += 1
        assert count_pos == 0

        return full_adj, flat_questions[:, 0], flat_questions[:, 1], gt, q_ids

    def setup_test_compatibility(self, resampled=False):
        """
        """
        comp_file = self.path_dataset + 'compatibility_test.json'
        if resampled:
            comp_file = self.path_dataset + 'compatibility_RESAMPLED_test.json'
        with open(comp_file) as f:
            self.comp_outfits = json.load(f)
