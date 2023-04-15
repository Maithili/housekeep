import sys
import pickle
from itertools import chain
from copy import deepcopy

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
    CLUSTERS = 4

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


def process_clusters():
    ''' Converts clusters of annotator labels to clusters of receptacle indices 
        for each object-room combination.'''

    clustered_object_preferences = pd.read_csv('./user_preferences_clustered_num-4_housekeep_poslessthan1en2.csv')

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

        print(f'{object_name} and {room_name}: {len(clusters_or)} clusters formed')

        all_clusters[f'{object_name}+{room_name}'] = deepcopy(clusters_or) # + for splitting object and room names

    with open('all_clusters_poslessthan1en2.pkl', 'wb') as fw:
        pickle.dump(all_clusters, fw)


def users_from_clusters():
    global housekeep_data
    global object2index, rooms2index

    raise NotImplementedError

    with open('all_clusters_poslessthan1en2.pkl', 'rb') as fh:
        cluster_objects_dict = pkl.load(fh)

    filtered_byroom_clusters = dict()

    for key, value in cluster_objects_dict.items():

        room = key.split('+')[-1]

        if room not in desired_rooms: continue

        filtered_byroom_clusters[key] = value

    filtered_keys = list(filtered_byroom_clusters.keys())

    data = []

    textfh = open('./log_housekeep_personas_poslessthan1en2.log', 'w')
    while len(data) < 5000:

        random_key_comb = np.random.choice(filtered_keys, size=10, replace=False)

        object_names = [r.split('+')[0] for r in random_key_comb]

        if any([object_names.count(f)>1 for f in object_names]): # objects should not repeat
            continue

        # choose random cluster per object-room combination
        random_key_clusters = \
            np.array([np.random.randint(0, len(cluster_objects_dict[k]), size=1)[0] for k in random_key_comb])
        
        # argsort the object names
        sort_indices = np.argsort(random_key_comb)
        
        flag_continue = False
        if len(data) > 0:
            for elem in data:
                if all(elem[0] == random_key_comb[sort_indices]): # if set of objects repeat
                    flag_continue = True
                    break

        if flag_continue:
            continue

        else:

            data.append((random_key_comb[sort_indices], random_key_clusters[sort_indices])) # add to data list

            for k, c in zip(random_key_comb[sort_indices], random_key_clusters[sort_indices]): # print
                textfh.write('{}: {}\n'.format(k, [room_receps[i] for i in cluster_objects_dict[k][c]]))
            textfh.write('---\n\n')

    textfh.close()

    print(f'len of data is {len(data)}')

    with open('housekeep_personas_poslessthan1en2.pkl', 'wb') as fw:
        pkl.dump(data, fw)

if __name__ == '__main__':

    if sys.argv[1] == 'cluster':
        cluster_annotators()
        process_clusters()

    elif sys.argv[1] == 'generate_data':
        users_from_clusters()

    else:
        raise NotImplementedError('Unknown argument: {}. Please use either `cluster` or `generate_data`.'.format(sys.argv[1]))