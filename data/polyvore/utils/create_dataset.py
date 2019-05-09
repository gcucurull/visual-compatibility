"""
Create the dataset files for Polyvore from the raw data.
"""

import os
import json
import scipy as sp
from scipy.sparse import lil_matrix, save_npz, csr_matrix
import argparse
import pickle as pkl
import numpy as np
from get_questions import get_questions
from get_compatibility import get_compats
from resample_fitb import resample_fitb
from resample_compat import resample_compatibility

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train', choices=['train', 'valid', 'test'])
args = parser.parse_args()

save_path = '../dataset/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

dataset_path = '../jsons/'
json_file = dataset_path + '{}_no_dup.json'.format(args.phase)
train_file = dataset_path + 'train_no_dup.json'
valid_file = dataset_path + 'valid_no_dup.json'
test_file = dataset_path + 'test_no_dup.json'
with open(json_file) as f:
    json_data = json.load(f)

# load the features extracted with 'extract_features.py'
feat_pkl = os.path.join(save_path, 'imgs_featdict_{}.pkl'.format(args.phase))
if os.path.exists(feat_pkl):
    with open(feat_pkl, 'rb') as handle:
        feat_dict = pkl.load(handle)
else:
    raise FileNotFound('The extracted features file {} does not exist'.format(feat_pkl))

relations = {}
id2idx = {}
idx = 0
features = []

# map an image id to their ids with format 'OUTFIT-ID_INDEX'
map_id2their = {}

for outfit in json_data:
    outfit_ids = set()
    for item in outfit['items']:
        # get id from the image url
        _, id = item['image'].split('id=')
        id = int(id)
        outfit_ids.add(id)
        #cat_dict[id] = item['categoryid']
        map_id2their[id] = '{}_{}'.format(outfit['set_id'], item['index'])

    for id in outfit_ids:
        if id not in relations:
            relations[id] = set()
            img_feats = feat_dict[str(id)] 
            # TODO, REMOVE
            #cat_vector = cat_vectors[cat_dict[id]]
            #feats = np.concatenate((cat_vector, img_feats))
            features.append(img_feats)
            # map this id to a sequential index
            id2idx[id] = idx
            idx += 1

        relations[id].update(outfit_ids)
        relations[id].remove(id)

map_file = save_path + 'id2idx_{}.json'.format(args.phase)
with open(map_file, 'w') as f:
    json.dump(id2idx, f)
map_file = save_path + 'id2their_{}.json'.format(args.phase)
with open(map_file, 'w') as f:
    json.dump(map_id2their, f)

# create sparse matrix that will represent the adj matrix
sp_adj = lil_matrix((idx, idx))
features_mat = np.zeros((idx, 2048))
print('Filling the values of the sparse adj matrix')
for rel in relations:
    rel_list = relations[rel]
    from_idx = id2idx[rel]
    features_mat[from_idx] = features[from_idx]

    for related in rel_list:
        to_idx = id2idx[related]

        sp_adj[from_idx, to_idx] = 1
        sp_adj[to_idx, from_idx] = 1 # because it is symmetric

print('Done!')

density = sp_adj.sum() / (sp_adj.shape[0] * sp_adj.shape[1])
print('Sparse density: {}'.format(density))

# now save the adj matrix
save_adj_file = save_path + 'adj_{}.npz'.format(args.phase)
sp_adj = sp_adj.tocsr()
save_npz(save_adj_file, sp_adj)

save_feat_file = save_path + 'features_{}'.format(args.phase)
sp_feat = csr_matrix(features_mat)
save_npz(save_feat_file, sp_feat)

def create_test(resampled=False):
    if resampled:
        resample_fitb()
        resample_compatibility()

    # build the question indexes
    questions = get_questions(resampled=resampled)
    for i in range(len(questions)): # for each question
        assert len(questions[i]) == 4
        for j in range(2): # questions list (j==0) or answers list (j==1)
            for z in range(len(questions[i][j])): # for each id in the list
                id = int(questions[i][j][z])
                questions[i][j][z] = id2idx[id] # map the id to the node index

    questions_file = save_path + 'questions_{}.json'.format(args.phase)
    if resampled:
        questions_file = questions_file.replace('questions', 'questions_RESAMPLED')
    with open(questions_file, 'w') as f:
        json.dump(questions, f)

    # outfit compat task
    outfits = get_compats(resampled=resampled)
    for i in range(len(outfits)): # for each outfit
        for j in range(len(outfits[i][0])):
            id = int(outfits[i][0][j])
            outfits[i][0][j] = id2idx[id]

    compat_file = save_path + 'compatibility_{}.json'.format(args.phase)
    if resampled:
        compat_file = compat_file.replace('compatibility', 'compatibility_RESAMPLED')
    print(compat_file)
    with open(compat_file, 'w') as f:
        json.dump(outfits, f)

if args.phase == 'test':
    create_test(False)
    create_test(True)
