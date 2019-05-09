"""
Resample the FITB choices so that they share category with the correct item.
"""

import json
import numpy as np
from random import shuffle
import random

def resample_fitb():
    dataset_path = '../jsons/'
    questions_file = dataset_path + 'fill_in_blank_test.json'
    json_file = dataset_path + '{}_no_dup.json'.format('test')
    questions_file_resampled = dataset_path + 'fill_in_blank_test_RESAMPLED.json'

    with open(json_file) as f:
        json_data = json.load(f)

    with open(questions_file) as f:
        questions_data = json.load(f)

    questions_resampled = []

    print('There are {} questions.'.format(len(questions_data)))
    print('There are {} test outfits.'.format(len(json_data)))

    # map ids in the form 'outfit_index' to 'imgID', because in 'fill_in_blank_test.json'
    # the ids are in format outfit_index but I use the other ID (so that the same item in two outfits has the same id)
    map_ids = {}
    map_cat = {}
    for outfit in json_data:
        outfit_ids = set()
        for item in outfit['items']:
            outfit_id = '{}_{}'.format(outfit['set_id'], str(item['index']))
            # get id from the image url
            _, id = item['image'].split('id=')
            map_ids[outfit_id] = id
            map_cat[outfit_id] = item['categoryid']

    not_enough = 0
    for ques in questions_data:
        new_ques = {'blank_position': ques['blank_position']}
        q = []
        for q_id in ques['question']:
            outfit_id = q_id.split('_')[0]
            #q_id = map_ids[q_id]
            q.append(q_id)
        new_ques['question'] = q

        a = []
        i = 0
        catid = -1
        for a_id in ques['answers']:
            if i == 0:
                # correct choice
                assert a_id.split('_')[0] == outfit_id
                catid = map_cat[a_id]
                choices = set([out_id for out_id in map_cat if map_cat[out_id] == catid]) - set([a_id]) - set(q)
                choices = list(choices)
                shuffle(choices)
            else: # resample item that has the category 'catid' (which is the same as the missing item))
                # it could happen that there aren't enough items of that category
                if i-1 < len(choices):
                    a_id = choices[i-1]
                else:
                    # not enough possible choices with the same cat
                    a_id = random.choice(list(map_cat.keys()))
                    not_enough += 1

            #a_id = map_ids[a_id]
            a.append(a_id)
            i += 1
        new_ques['answers'] = a

        questions_resampled.append(new_ques)

    print(not_enough)

    with open(questions_file_resampled, 'w') as f:
        json.dump(questions_resampled, f)

    print('Saved!')
