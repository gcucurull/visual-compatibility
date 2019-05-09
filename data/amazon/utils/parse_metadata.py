import gzip
import os
import pickle as pkl

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

data_path = '../'
meta_file = os.path.join(data_path, 'meta_Clothing_Shoes_and_Jewelry.json.gz')
metadata = {}
print('Parsin metadata...')
for i,l in enumerate(parse(meta_file)):
	metadata[l['asin']] = l

metadata_path = os.path.join(data_path, 'metadata.pkl')
with open(metadata_path, 'wb') as f:
    pkl.dump(metadata, f)