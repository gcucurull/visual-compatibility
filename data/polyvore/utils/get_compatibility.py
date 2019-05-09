import json

def get_compats(resampled=False):
    dataset_path = '../jsons/'

    #################
    #### MAP IDS ####
    #################

    json_file = dataset_path + '{}_no_dup.json'.format('test')
    with open(json_file) as f:
        json_data = json.load(f)

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

    ################################
    #### PROCESS COMPAT OUTFITS ####
    ################################

    if resampled:
        compat_file = dataset_path + 'fashion_compatibility_prediction_RESAMPLED.txt'
    else:
        compat_file = dataset_path + 'fashion_compatibility_prediction.txt'

    outfits = []
    n_comps = 0
    with open(compat_file) as f:
        for line in f:
            cols = line.rstrip().split(' ')
            compat_score = float(cols[0])
            assert compat_score in [1, 0]
            items = cols[1:]
            # map their ids to my img ids
            items = [map_ids[it] for it in items]
            n_comps += 1
            outfits.append((items, compat_score))

    print('There are {} outfits to test compatibility'.format(n_comps))
    print('There are {} test outfits.'.format(len(json_data)))

    # returns 2 lists:
    # - outfits: len N, contains lists of outfits
    # - labels: len N, contains the labels corresponding to the outfits
    return outfits

def main():
    compatibilities = get_compats()

if __name__ == '__main__':
    main()
