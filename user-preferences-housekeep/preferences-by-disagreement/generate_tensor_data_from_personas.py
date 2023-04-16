import os
import sys
import pickle as pkl

'''
from transformers import AutoTokenizer, CLIPTextModel
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
outputs = model(**inputs)
'''
import torch
import torch.nn.functional as F
import numpy as np


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
    train_data_dict = dict()

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

                train_data_dict[f'u{pid}-d{data_id}-seen'] = dict({
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

                train_data_dict[f'u{pid}-d{data_id}-seen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'recept_name': neg_receptname,
                    'room_name': neg_roomname, 
                    'room_embb': room_encoding_matrix[rooms2index[neg_roomname]], 
                    'user_embb': user_embb, 
                    'ground_truth_score': 0
                    })

                data_id += 1

    return train_data_dict


def generate_test_data(persona_data):
    global rooms2index, roomrecepts2index    

    global room_recepts_all
    global user_encoding_matrix
    global room_encoding_matrix

    data_id = 0
    test_data_dict = dict()

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

                test_data_dict[f'u{pid}-d{data_id}-unseen'] = dict({
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

                test_data_dict[f'u{pid}-d{data_id}-unseen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'recept_name': neg_receptname,
                    'room_name': neg_roomname, 
                    'room_embb': room_encoding_matrix[rooms2index[neg_roomname]], 
                    'user_embb': user_embb, 
                    'ground_truth_score': 0
                    })

                data_id += 1

    return test_data_dict


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

    train_data_dict = generate_train_data(persona_data)
    test_data_dict = generate_test_data(persona_data)


if __name__ == '__main__':

    np.random.seed(8213546)
    torch.manual_seed(8213546)

    if sys.argv[1] == 'save_tensor':
        main()

    else:
        raise NotImplementedError('Unknown argument: {}. Please use `save_tensor`.'.format(sys.argv[1]))