import array
import numpy as np
import time
import gzip
import scipy.sparse as sp
import os
import pickle as pkl

def NEWreadImageFeatures(path):
    f = open(path, 'rb')
    while True:
        asin = f.read(10)
        if not asin:
            break
        a = array.array('f')
        a.frombytes(f.read(4096*4))
        yield asin, a.tolist()
    f.close()

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)    

data_path = '../'

# first select products
meta_path = data_path + 'meta_Clothing_Shoes_and_Jewelry.json.gz'
filtered_cat = {} # split products in subcategories
cats_of_interest = ['Women', 'Men']
all_prods = {}

valid = 0
before = time.time()
for i,l in enumerate(parse(meta_path)):
    if 'related' not in l.keys():
        continue
        
    if 'also_bought' not in l['related'] or 'bought_together' not in l['related']:
        continue
        
    categories = l['categories'][0]
    if categories[0] != 'Clothing, Shoes & Jewelry':
        continue
        
    second_cat = categories[1]
    if second_cat not in cats_of_interest:
        continue
    
    if second_cat not in filtered_cat.keys():
        filtered_cat[second_cat] = {}
    filtered_cat[second_cat][l['asin']] = l
    all_prods[l['asin']] = l
    
    valid += 1
    
print("Processed: {}, valid: {}. Elapsed: {}s".format(i, valid, time.time() - before))

for cat in filtered_cat:
    print(cat, len(filtered_cat[cat]))

# read product features
before = time.time()
prod_features = {}
i = 0
feats_path = data_path + 'image_features_Clothing_Shoes_and_Jewelry.b'
for asin, feat in NEWreadImageFeatures(feats_path):
    prod_id = asin.decode('utf-8')
    if prod_id in all_prods:
        prod_features[prod_id] = np.array(feat)
        i += 1
        if i % 50000 == 0:
            elapsed = time.time() - before
            avg = elapsed / i
            print(i, avg)

print('Done!')

def build_adj(category='Men', relation='bought_together'):
    print('building data structures...')
    before = time.time()
    holder = {}
    print('Cat: {}, rel: {}'.format(category, relation))
    prod_dict = filtered_cat[category]
    edges = []
    for prod in prod_dict:
        if prod in prod_features: # check there exist features for this product
            title = prod_dict[prod]['title'] if 'title' in prod_dict[prod] else None
            bought_together = [rel for rel in prod_dict[prod]['related'][relation] if rel in prod_dict]

            for rel in bought_together:
                if rel in prod_features: # check there exist features for this product
                    edges.append((prod, rel))

    print(" - There are {} edges".format(len(edges)))

    id2idx = {} # map amazon ids to nodes indexes
    idx2id = {} # map node indexes to amazon ids
    count_unique = 0
    for edge in edges:
        for id in edge:
            if id not in id2idx:
                id2idx[id] = count_unique
                idx2id[count_unique] = id
                count_unique += 1

    print(" - There are {} unique nodes".format(count_unique))

    adj = sp.lil_matrix((count_unique, count_unique))
    for edge in edges:
        u, v = edge
        u_idx = id2idx[u]
        v_idx = id2idx[v]
        adj[u_idx, v_idx] = 1
        adj[v_idx, u_idx] = 1
    adj.setdiag(0)

    print(' - adj sum /2:', int(adj.sum()/2))
    print(' - avg degree: ', adj.sum(axis=1).mean())
    
    holder['adj'] = adj.tocsr()
    holder['id2idx'] = id2idx
    holder['idx2id'] = idx2id
    
    # fill the features matrix
    feats_mat = np.zeros((count_unique, 4096))
    
    keys = prod_features.keys()
    for jj,asin in enumerate(id2idx.keys()):
        idx = id2idx[asin]
        if asin in keys:
            feats_mat[idx, :] = prod_features[asin][:]

    sumf = feats_mat.sum(axis=1)
    assert (sumf == 0).sum() == 0
    
    holder['feats_mat'] = feats_mat
    
    elapsed = time.time() - before
    print('Elapsed', elapsed)
    
    return holder

def create(cat, rel):
    holder = build_adj(cat, rel)
    folder = '{}_{}/'.format(cat, rel)

    print(holder['adj'].shape)

    base_path = '../dataset/'
    folder_path = os.path.join(base_path , folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    feat_mat_path = os.path.join(folder_path, 'feats.npy')
    np.save(feat_mat_path, holder['feats_mat'])

    adj_path = os.path.join(folder_path, 'adj.npz')
    sp.save_npz(adj_path, holder['adj'])

    id2idx_path = os.path.join(folder_path, 'id2idx.pkl')
    idx2id_path = os.path.join(folder_path, 'idx2id.pkl')
    with open(id2idx_path, 'wb') as f:
        pkl.dump(holder['id2idx'], f)
    with open(idx2id_path, 'wb') as f:
        pkl.dump(holder['idx2id'], f)

    print('done!')

for cat in ['Men', 'Women']:
    for rel in ["bought_together", "also_bought"]:
        create(cat, rel)
