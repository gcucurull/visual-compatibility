"""
The invalid outfits for the compat task are very easy, since they don't need to
form a valid outfit. That's why we resample the invalid outfits, so that they have
the same categories as the valid ones, making them be invalid only because of their
style.
"""

import json
import os
import numpy as np

def resample_compatibility():
    data_path = '../jsons/'
    orig_file = os.path.join(data_path, 'fashion_compatibility_prediction.txt')
    dest_file = os.path.join(data_path, 'fashion_compatibility_prediction_RESAMPLED.txt')

    valid_outfits = []
    invalid_outfits = []
    with open(orig_file) as f:
        for line in f:
            res = line.rstrip().split(' ')
            score, items = int(res[0]), res[1:]
            if score:
                valid_outfits.append(items)
            else:
                invalid_outfits.append(items)

    json_file = os.path.join(data_path, 'test_no_dup.json')
    with open(json_file) as f:
        json_data = json.load(f)

    item_cats = {} # map item to categories
    cat_items = {} # list of items for each category
    for outfit in json_data:
        set_id = outfit['set_id']
        for item in outfit['items']:
            idx = item['index']
            item_id = '{}_{}'.format(set_id, idx)
            item_cats[item_id] = item['categoryid']
            if item['categoryid'] not in cat_items:
                cat_items[item['categoryid']] = []
            cat_items[item['categoryid']].append(item_id)

    new_invalid = []
    for i in range(len(invalid_outfits)):
        new_out = []
        base = np.random.choice(valid_outfits) # choose a valid outfit to copy its categories
        for item in base:
            cat = item_cats[item]
            # sample an item with the same category
            new_item = np.random.choice(cat_items[cat])
            new_out.append(new_item)
        new_invalid.append(new_out)

    with open(dest_file, 'w') as f:
        for out in valid_outfits:
            write_str = '1'
            for item in out:
                write_str += ' {}'.format(item)
            write_str += '\n'
            f.write(write_str)
        for out in new_invalid:
            write_str = '0'
            for item in out:
                write_str += ' {}'.format(item)
            write_str += '\n'
            f.write(write_str)
