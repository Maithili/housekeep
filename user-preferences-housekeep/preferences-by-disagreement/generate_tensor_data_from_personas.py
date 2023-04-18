import os
import sys
import pickle as pkl
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel


## GLOBAL VARIABLES -----------------------------

npy_data = np.load('./housekeep.npy', allow_pickle=True).item()

object2index = dict({v:int(k) for k, v in enumerate(npy_data['objects'])})
rooms2index = dict({v:int(k) for k, v in enumerate(npy_data['rooms'])})
roomrecepts2index = dict({v:int(k) for k, v in enumerate(npy_data['room_receptacles'])})

housekeep_data = npy_data['data']


def generate_train_data(persona_data):
    global rooms2index, roomrecepts2index    

    global room_recepts_all
    global user_encoding_matrix
    global room_encoding_matrix

    data_id = 0
    train_pos_data_dict = dict()
    train_neg_data_dict = dict()

    for pid in range(len(persona_data)):

        user_embb = user_encoding_matrix[pid, :]

        # positive room-receptacle pairs for user [pid]
        for skey, srecept_id_list in zip(persona_data[pid]['seen_keys'], 
                                        persona_data[pid]['seen_key_recepts']):
            
            object_name, room_name = skey.split('+')
            room_embb = room_encoding_matrix[rooms2index[room_name], :]

            for srecept_id in srecept_id_list:

                srecept_fullname = room_recepts_all[srecept_id]
                srecept_roomname, srecept_receptname = \
                    srecept_fullname.split('|')

                if not srecept_roomname == room_name: #TODO: why is this even happening 
                    print(f'Seen (user {pid}, key {skey}): {srecept_roomname} does not match with {room_name}')
                    continue # skip this example 

                train_pos_data_dict[f'u{pid}-d{data_id}-seen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'recept_name': srecept_receptname,
                    'room_name': room_name, 
                    'room_embb': room_embb, 
                    'user_embb': user_embb, 
                    'ground_truth_score': 1
                    })

                data_id += 1

            # sample same number of negative room-receptacle pairs for this user
            num_pos_samples = len(srecept_id_list)

            neg_room_recepts_list = [r for r in room_recepts_all 
                                        if roomrecepts2index[r] not in srecept_id_list]

            np.random.shuffle(neg_room_recepts_list)

            for neg_room_recept in neg_room_recepts_list[:num_pos_samples]:

                neg_roomname, neg_receptname = neg_room_recept.split('|')

                train_neg_data_dict[f'u{pid}-d{data_id}-seen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'recept_name': neg_receptname,
                    'room_name': neg_roomname, 
                    'room_embb': room_encoding_matrix[rooms2index[neg_roomname]], 
                    'user_embb': user_embb, 
                    'ground_truth_score': 0
                    })

                data_id += 1

    return train_pos_data_dict, train_neg_data_dict


def generate_test_data(persona_data):
    global rooms2index, roomrecepts2index    

    global room_recepts_all
    global user_encoding_matrix
    global room_encoding_matrix

    data_id = 0
    test_pos_data_dict = dict()
    test_neg_data_dict = dict()

    for pid in range(len(persona_data)):

        user_embb = user_encoding_matrix[pid, :]

        # positive room-receptacle pairs for user [pid]
        for uskey, usrecept_id_list in zip(persona_data[pid]['unseen_keys'], 
                                        persona_data[pid]['unseen_key_recepts']):
            
            object_name, room_name = uskey.split('+')
            room_embb = room_encoding_matrix[rooms2index[room_name], :]

            for usrecept_id in usrecept_id_list:

                usrecept_fullname = room_recepts_all[usrecept_id]
                usrecept_roomname, usrecept_receptname = \
                    usrecept_fullname.split('|')

                if not usrecept_roomname == room_name: #TODO: why is this even happening 
                    print(f'Unseen (user {pid}, key {uskey}): {usrecept_roomname} does not match with {room_name}')
                    continue # skip this example 

                test_pos_data_dict[f'u{pid}-d{data_id}-unseen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'recept_name': usrecept_receptname,
                    'room_name': room_name, 
                    'room_embb': room_embb, 
                    'user_embb': user_embb, 
                    'ground_truth_score': 1
                    })

                data_id += 1

            # sample same number of negative room-receptacle pairs for this user
            num_pos_samples = len(usrecept_id_list)

            neg_room_recepts_list = [r for r in room_recepts_all 
                                        if roomrecepts2index[r] not in usrecept_id_list]

            np.random.shuffle(neg_room_recepts_list)

            for neg_room_recept in neg_room_recepts_list[:num_pos_samples]:

                neg_roomname, neg_receptname = neg_room_recept.split('|')

                test_neg_data_dict[f'u{pid}-d{data_id}-unseen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'recept_name': neg_receptname,
                    'room_name': neg_roomname, 
                    'room_embb': room_encoding_matrix[rooms2index[neg_roomname]], 
                    'user_embb': user_embb, 
                    'ground_truth_score': 0
                    })

                data_id += 1

    return test_pos_data_dict, test_neg_data_dict


def tensor_data_with_clip(data_dict, model, tokenizer):

    keys_list = list(data_dict.keys())

    # inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
    # outputs = model(**inputs)

    objects_list = []
    recepts_list = []

    for key in keys_list:
        objects_list.append(data_dict[key]['object_name'].replace('_', ' '))
        recepts_list.append(data_dict[key]['recept_name'].replace('_', ' '))

    # print(objects_list)
    # print(recepts_list)

    objects_tokens = tokenizer(objects_list, padding=True, return_tensors="pt")
    objects_embbs = model(**objects_tokens).pooler_output

    recepts_tokens = tokenizer(recepts_list, padding=True, return_tensors="pt")
    recepts_embbs = model(**recepts_tokens).pooler_output

    updated_data_dict = deepcopy(data_dict)

    print(objects_embbs.shape)
    print(recepts_embbs.shape)

    for i, key in enumerate(keys_list):

        updated_data_dict[key]['object_embb'] = objects_embbs[i, :]
        updated_data_dict[key]['recept_embb'] = recepts_embbs[i, :]

    return updated_data_dict

def main():
    global npy_data
    global rooms2index, roomrecepts2index

    global room_recepts_all
    global user_encoding_matrix
    global room_encoding_matrix

    rooms_all = npy_data['rooms']
    room_recepts_all = npy_data['room_receptacles']
    
    # load persona data
    with open('housekeep_personas_poslessthan1en2_maxclusters3.pkl', 'rb') as fh:
        persona_data_dict = pkl.load(fh)

    persona_data = persona_data_dict['personas']
    print(persona_data[0]['seen_keys'])
    print(persona_data[0]['seen_key_recepts'])
    print(persona_data[0]['unseen_keys'])
    print(persona_data[0]['unseen_key_recepts'])

    # representation = {clip for object, clip for receptacle, 
    #                       one hot for room, one hot for user}
    # output = score 0 to 1
    # questions: sampling negative examples? test on negative examples?

    user_encoding_matrix = F.one_hot(
        torch.arange(0, int(persona_data_dict['num_users'])))
    room_encoding_matrix = F.one_hot(
        torch.arange(0, len(rooms_all)))

    train_pos_data_dict, train_neg_data_dict = generate_train_data(persona_data)
    test_pos_data_dict, test_neg_data_dict = generate_test_data(persona_data)

    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    train_pos_tensor_data = tensor_data_with_clip(train_pos_data_dict, model, tokenizer)
    train_neg_tensor_data = tensor_data_with_clip(train_neg_data_dict, model, tokenizer)
    test_pos_tensor_data = tensor_data_with_clip(test_pos_data_dict, model, tokenizer)
    test_neg_tensor_data = tensor_data_with_clip(test_neg_data_dict, model, tokenizer)

    torch.save(dict({'train-pos': train_pos_tensor_data, 
                     'train-neg': train_neg_tensor_data, 
                     'test-pos': test_pos_tensor_data, 
                     'test-neg': test_neg_tensor_data}), 'personas_tensor_data.pt')


if __name__ == '__main__':

    np.random.seed(8213546)
    torch.manual_seed(8213546)

    if sys.argv[1] == 'save_tensor':
        main()

    else:
        raise NotImplementedError('Unknown argument: {}. Please use `save_tensor`.'.format(sys.argv[1]))