import json
import numpy as np

def get_questions(resampled=False):
    dataset_path = '../jsons/'
    if resampled:
        questions_file = dataset_path + 'fill_in_blank_test_RESAMPLED.json'
    else:
        questions_file = dataset_path + 'fill_in_blank_test.json'
    json_file = dataset_path + '{}_no_dup.json'.format('test')

    with open(json_file) as f:
        json_data = json.load(f)

    with open(questions_file) as f:
        questions_data = json.load(f)

    questions_save = []

    print('There are {} questions.'.format(len(questions_data)))
    print('There are {} test outfits.'.format(len(json_data)))

    # map ids in the form 'outfit_index' to 'imgID', because in 'fill_in_blank_test.json'
    # the ids are in format outfit_index but I use the other ID (so that the same item in two outfits has the same id)
    map_ids = {}
    for outfit in json_data:
        outfit_ids = set()
        for item in outfit['items']:
            outfit_id = '{}_{}'.format(outfit['set_id'], str(item['index']))
            # get id from the image url
            _, id = item['image'].split('id=')
            map_ids[outfit_id] = id

    save_data = []
    counts = 0
    count_all_valid = 0
    for ques in questions_data:
        q = []
        for q_id in ques['question']:
            outfit_id = q_id.split('_')[0]
            q_id = map_ids[q_id]
            q.append(q_id)
        a = []
        positions = []
        i = 0
        for a_id in ques['answers']:
            if i == 0:
                assert a_id.split('_')[0] == outfit_id
            else:
                if a_id.split('_')[0] == outfit_id:
                    pass # this is true for a few edge queries
            pos = int(a_id.split('_')[-1]) # get the posittion of this item within the outfit
            a_id = map_ids[a_id]
            a.append(a_id)
            positions.append(pos)
            i += 1

        # count how many questions have only one possible choice of the correct category
        choices = sum([p == ques['blank_position'] for p in positions])
        counts += choices == 1
        count_all_valid += choices == 4
        save_data.append([q, a, positions, ques['blank_position']])

    # save_data is a list of questions
    # each question is a list that contains:
    #    - list of outfit IDs (len N)
    #    - list of possible answers (len 4)
    #    - list of possible answers positions
    #    - desired position
    return save_data

def main():
    questions = get_questions()

if __name__ == '__main__':
    main()
