import sys
import pickle
from itertools import chain
from copy import copy, deepcopy

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.cluster import AgglomerativeClustering

## GLOBAL VARIABLES -----------------------------

npy_data = np.load('./housekeep.npy', allow_pickle=True).item()

object2index = dict({v:k for k, v in enumerate(npy_data['objects'])})
rooms2index = dict({v:k for k, v in enumerate(npy_data['rooms'])})

housekeep_data = npy_data['data']

## UTILS -----------------------------


def receptacle_labels2vec(x):
    ''' Converts list of receptacle indices to a vector of 0s and 1s. '''   

    result = np.zeros((128,)) # because of 128 receptacles
    for recep in x:
        result[recep] = 1
    return result


def distance_metric(x, y):

    if np.all(x==y): # exactly same
        return 0

    dot_prod = np.dot(x, y)

    if (sum(x**2) + sum(y**2) - dot_prod) == 0:
        print(sum(x), sum(y))
        input('wait')

    # jaccard distance = 1 - jaccard similarity = 1 - (a n b)/(a U b)
    return 1 - dot_prod/(sum(x**2) + sum(y**2) - dot_prod)

## SCRIPTS -----------------------------


def cluster_annotators():
    ''' Clusters annotator labels by representing correct receptacles 
        as a vector and performing agglomerative clustering.'''

    global npy_data, housekeep_data

    objects = npy_data['objects']
    rooms = npy_data['rooms']

    # hyperparameters
    CLUSTERS = 3

    clustering_pandas_df = pd.DataFrame([], 
        columns=['object', 'room', 'identifiers', 'cluster_asgns'])

    disagreed_object_names = []

    # load objects from file
    with open('housekeep_poslessthan1en2_agreement.txt', 'r') as fh:
        alllines = fh.readlines()

        for line in alllines:
            disagreed_object_names.append(line.split(',')[0])

    for object_idx in range(268):
        object_name = objects[object_idx]

        for room_idx in range(17):

            room_name = rooms[room_idx]

            if '{}/{}'.format(object_name, room_name) not in disagreed_object_names:
                continue

            print(f'obj: {object_name}, room: {room_name}') #DEBUG

            filtered_values = np.where((housekeep_data['object_idx']==object_idx)&(housekeep_data['room_idx']==room_idx))
            identifiers = []
            datapoints = []

            # lists to vector
            for i, d in housekeep_data.loc[filtered_values].iterrows():

                vec = receptacle_labels2vec(d['correct'])

                identifiers.append('ann{}_asgn{}'.format(d['annotator_idx'], d['assignment_idx']))
                datapoints.append(vec[np.newaxis, :])

            assert len(identifiers) == 10

            # array of datapoints
            datapoints_array = np.concatenate(datapoints, axis=0)

            # pairwise distance matrix
            pairwise_dist_mat = np.zeros((len(datapoints_array), len(datapoints_array)))
            for i1, d1 in enumerate(datapoints_array):
                for i2, d2 in enumerate(datapoints_array):
                    pairwise_dist_mat[i1, i2] = distance_metric(d1, d2)

            # clusters
            num_clusters = CLUSTERS

            clusters = AgglomerativeClustering(n_clusters=num_clusters, affinity="precomputed", linkage="single").fit_predict(pairwise_dist_mat)
            unique_cluster_asgns, cluster_counts = np.unique(clusters, return_counts=True)

            if any([x<2 for x in cluster_counts]): # delete examples with only one annotator in a cluster
                print(f'{object_name}-{room_name}: no clustering!')
                continue

            else:
                print('Cluster count: ', cluster_counts)

            # save data to pandas file
            clustering_pandas_df = pd.concat([clustering_pandas_df, 
                                              pd.DataFrame(dict({'object':[object_name],
                                                                    'room': [room_name],
                                                                    'identifiers': (len(identifiers)*'{}-').format(*identifiers),
                                                                    'cluster_asgns': (len(clusters)*'{}-').format(*clusters)
                                                                    }))
                                                ], ignore_index=True)

        # break # DEBUG for test purposes

    print('done')

    # save
    clustering_pandas_df.to_csv(f'./user_preferences_clustered_num-{CLUSTERS}_housekeep_poslessthan1en2.csv')
    print('saved')
    print(clustering_pandas_df.head(-5))


def process_clusters(max_num_clusters):
    ''' Converts clusters of annotator labels to clusters of receptacle indices 
        for each object-room combination.'''

    clustered_object_preferences = pd.read_csv(f'./user_preferences_clustered_num-{max_num_clusters}_housekeep_poslessthan1en2.csv')

    all_clusters = dict()

    for _, row in clustered_object_preferences.iterrows():

        object_name = row['object']
        room_name = row['room']

        annotators = [int(x.split('_')[0][3:]) for x in row['identifiers'].split('-') if x != ''] # annotator_idx
        assignment_ids = [int(x.split('_')[1][4:]) for x in row['identifiers'].split('-') if x != ''] # assignment_idx
        clustering_asgns = [int(y) for y in row['cluster_asgns'].split('-') if y != '']

        clusters_or = []

        for c in np.unique(clustering_asgns): # actually cluster the items
            indexes = np.where(clustering_asgns==c)[0]

            correct = [housekeep_data[(housekeep_data['object_idx'] == object2index[object_name]) &
                                        (housekeep_data['annotator_idx'] == annotators[i]) &
                                        (housekeep_data['assignment_idx'] == assignment_ids[i])]['correct'].tolist()[0]
                        for i in indexes]
            correct_combined = sorted(set(chain.from_iterable(correct)))

            if len(correct_combined) == 0: # if annotators did not mark any correct receptacle, skip
                continue

            if len(clusters_or) > 1:
                if all([correct_combined != c for c in clusters_or]): # if cluster repeats itself
                    clusters_or.append(correct_combined)

            else:
                clusters_or.append(correct_combined)

        if len(clusters_or) == 0: continue 

        print(f'{object_name} and {room_name}: {len(clusters_or)} clusters formed')

        # if len(clusters_or) != 3: #DEBUG
        #     print(clusters_or)
        #     print(assignment_ids)
        #     input('wait')

        all_clusters[f'{object_name}+{room_name}'] = deepcopy(clusters_or) # "+" for splitting object and room names

    with open(f'all_clusters_poslessthan1en2_maxclusters{max_num_clusters}.pkl', 'wb') as fw:
        pickle.dump(all_clusters, fw)


def users_from_clusters():
    global housekeep_data
    global object2index, rooms2index

    # read list of seen objects
    with open('./housekeep_seen_objects.txt', 'r') as fh:
        output = fh.read()
        seen_objects_hkeep = [o.strip() for o in output.split(',')]

    # replace whitespace in items in seen_objects_hkeep with hyphen
    seen_objects_hkeep = [x.replace(' ', '_') for x in seen_objects_hkeep]
    seen_objects_hkeep = [x.replace('-', '_') for x in seen_objects_hkeep]


    # load clusters of correct receptacle labels
    with open('all_clusters_poslessthan1en2_maxclusters3.pkl', 'rb') as fh:
        cluster_objects_dict = pkl.load(fh)

    # Note: cluster_objects_dict = [obj-room pair] X [num_recept_clusters] X [correct recept labels]

    # list of all preference objects
    clustered_objs_list_all = list(set([k.split('+')[0] for k in cluster_objects_dict.keys()]))
    print('all clustered objects: ', len(clustered_objs_list_all))

    # split preference objects into seen and unseen
    clustered_objs_list_seen = [o for o in clustered_objs_list_all if o in seen_objects_hkeep]
    clustered_objs_list_unseen = [o for o in clustered_objs_list_all if o not in seen_objects_hkeep]

    # #DEBUG
    # print('all seen objects in hkeep: ', len(seen_objects_hkeep))
    # print('total number of obj-room preference pairs: ', len(cluster_objects_dict.keys()))
    # print(f'seen {len(clustered_objs_list_seen)}')
    # print(f'unseen {len(clustered_objs_list_unseen)}')

    user_personas = []

    while len(user_personas) < 15:

        print('Num of user personas finished: ', len(user_personas))

        # [10] random seen objects
        random_obj_comb_seen = np.random.choice(clustered_objs_list_seen, size=10, replace=False)

        matching_keys_randobj = dict({o: [k for k in cluster_objects_dict.keys() 
                                            if k.split('+')[0] == o and len(cluster_objects_dict[k])>=1] 
                                        for o in random_obj_comb_seen})

        assert not any([len(matching_keys_randobj[o]) == 0 for o in random_obj_comb_seen]) #

        if len(matching_keys_randobj) < 10: continue # minimum 10 seen objects

        random_keys_seen = [np.random.choice(matching_keys_randobj[o]) for o in random_obj_comb_seen]

        random_objs_unseen = []
        random_keys_unseen = []
        recepts_seen = []
        recepts_unseen = []

        for k_seen in random_keys_seen:
            found_match = False

            for o_unseen in clustered_objs_list_unseen:

                if found_match: break

                if o_unseen in random_objs_unseen: continue # cannot repeat unseen object

                for k_unseen in [k for k in cluster_objects_dict.keys() if k.split('+')[0] == o_unseen]:

                    if k_unseen in random_keys_unseen: continue # cannot repeat unseen key

                    # find rand cluster for o_seen
                    if len(cluster_objects_dict[k_seen]) > 1:
                        random_cluster_seen = np.random.randint(len(cluster_objects_dict[k_seen])-1)

                    elif len(cluster_objects_dict[k_seen]) == 1:
                        random_cluster_seen = 0

                    else:
                        raise KeyboardInterrupt

                    # find rand cluster for o_unseen
                    if len(cluster_objects_dict[k_unseen]) > 1:
                        random_cluster_unseen = np.random.randint(len(cluster_objects_dict[k_unseen])-1)

                    elif len(cluster_objects_dict[k_unseen]) == 1:
                        random_cluster_unseen = 0

                    else:
                        continue

                    # recept labels for o_seen and o_unseen clusters
                    recepts_seen_key_cluster = set(cluster_objects_dict[k_seen][random_cluster_seen])
                    recepts_unseen_key_cluster = set(cluster_objects_dict[k_unseen][random_cluster_unseen])

                    if len(recepts_seen_key_cluster.intersection(recepts_unseen_key_cluster)) > 2: # three or more

                        # print('seen recepts--', recepts_seen_key_cluster)
                        # print('unseen recepts--', recepts_unseen_key_cluster)

                        recepts_seen.append(recepts_seen_key_cluster)
                        recepts_unseen.append(recepts_unseen_key_cluster)

                        random_objs_unseen.append(o_unseen)
                        random_keys_unseen.append(k_unseen)

                        found_match = True

                        break

        compare_users = lambda x, y: sorted(x['seen_keys']) == sorted(y['seen_keys']) and sorted(x['unseen_keys']) == sorted(y['unseen_keys'])

        if len(random_keys_seen) == len(random_keys_unseen):
            user_x = dict({
                'seen_keys': copy(random_keys_seen),
                'seen_key_recepts': copy(recepts_seen),
                'unseen_keys': copy(random_keys_unseen),
                'unseen_key_recepts': copy(recepts_unseen),
            })

            if len(user_personas) == 0:
                user_personas.append(user_x)

            elif not any([compare_users(user_x, user_y) for user_y in user_personas]):
                user_personas.append(user_x)
                #TODO: delete clusters from original list

            else:
                continue

    for i, user_x in enumerate(user_personas):
        for user_y in user_personas[i+1:]:
            assert not compare_users(user_x, user_y)

    persona_data_dict = dict({
        'num_users': 15,
        'clusters_file': 'all_clusters_poslessthan1en2_maxclusters3.pkl',
        'personas': user_personas,
        })

    assert persona_data_dict['num_users'] == len(persona_data_dict['personas'])

    with open('housekeep_personas_poslessthan1en2_maxclusters3.pkl', 'wb') as fw:
        pkl.dump(persona_data_dict, fw)


if __name__ == '__main__':

    np.random.seed(8213546)

    if sys.argv[1] == 'cluster':
        # cluster_annotators()
        process_clusters(max_num_clusters=3)

    elif sys.argv[1] == 'generate_data':
        users_from_clusters()

    else:
        raise NotImplementedError('Unknown argument: {}. Please use either `cluster` or `generate_data`.'.format(sys.argv[1]))