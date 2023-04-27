import os
import sys
import pickle as pkl
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertModel, AutoTokenizer

## GLOBAL VARIABLES -----------------------------

npy_data = np.load('./housekeep.npy', allow_pickle=True).item()

object2index = dict({v:int(k) for k, v in enumerate(npy_data['objects'])})
rooms2index = dict({v:int(k) for k, v in enumerate(npy_data['rooms'])})
roomrecepts2index = dict({v:int(k) for k, v in enumerate(npy_data['room_receptacles'])})

housekeep_data = npy_data['data']


class DistillBERTEmbeddingGenerator():
    def __init__ (self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.map = {'objects':{}, 'recepts':{}}

    def __call__(self, text, token_type):

        assert token_type in self.map.keys()

        if text in self.map[token_type].keys(): # embb already computed
            return 

        assert '_' not in text, f'Underscore in text: {text}'

        inputs = self.tokenizer(text=text, return_tensors="pt")
        outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state[0,1:-1,:]
        last_hidden_states = last_hidden_states.mean(dim=0)
        assert last_hidden_states.size() == torch.Size([768])

        self.map[token_type][text] = last_hidden_states.detach()
    

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

                train_pos_data_dict[f'd{data_id}-seen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'seen_object': True,
                    'recept_name': srecept_receptname,
                    'room_name': room_name, 
                    'room_embb': room_embb, 
                    'user_id': pid, 
                    'user_embb': user_embb, 
                    'is_train': bool((data_id+1)%2), # if data_id is odd, then it is train data
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

                train_neg_data_dict[f'd{data_id}-seen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'seen_object': True,
                    'recept_name': neg_receptname,
                    'room_name': neg_roomname, 
                    'room_embb': room_encoding_matrix[rooms2index[neg_roomname]], 
                    'user_id': pid, 
                    'user_embb': user_embb, 
                    'is_train': bool((data_id+1)%2), # if data_id is odd, then it is train data
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

                test_pos_data_dict[f'd{data_id}-unseen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'seen_object': False,
                    'recept_name': usrecept_receptname,
                    'room_name': room_name, 
                    'room_embb': room_embb,
                    'user_id': pid, 
                    'user_embb': user_embb, 
                    'is_train': False, 
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

                test_neg_data_dict[f'd{data_id}-unseen'] = dict({
                    'data_id': data_id,
                    'object_name': object_name, 
                    'seen_object': False,
                    'recept_name': neg_receptname,
                    'room_name': neg_roomname, 
                    'room_embb': room_encoding_matrix[rooms2index[neg_roomname]], 
                    'user_id': pid, 
                    'user_embb': user_embb, 
                    'is_train': False, 
                    'ground_truth_score': 0
                    })

                data_id += 1

    return test_pos_data_dict, test_neg_data_dict


def tensor_data_with_bert(data_dict, bert_generator):

    keys_list = list(data_dict.keys())

    # inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
    # outputs = model(**inputs)

    objects_list = []
    recepts_list = []

    for key in keys_list:
        object_name = data_dict[key]['object_name'].replace('_', ' ')
        bert_generator(object_name, 'objects')
        objects_list.append(object_name)
        
        recept_name = data_dict[key]['recept_name'].replace('_', ' ')
        bert_generator(recept_name, 'recepts')
        recepts_list.append(recept_name)

    print('Objects: ', len(bert_generator.map['objects']))
    print('Recepts: ', len(bert_generator.map['recepts']))

    updated_data_dict = deepcopy(data_dict)

    for i, key in enumerate(keys_list):

        object_name = data_dict[key]['object_name'].replace('_', ' ')
        updated_data_dict[key]['object_embb'] = bert_generator.map['objects'][object_name]

        recept_name = data_dict[key]['recept_name'].replace('_', ' ')
        updated_data_dict[key]['recept_embb'] = bert_generator.map['recepts'][recept_name]

    return updated_data_dict

def main(persona_data_path):
    global npy_data
    global rooms2index, roomrecepts2index

    global room_recepts_all
    global user_encoding_matrix
    global room_encoding_matrix

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%m-%Y_%H-%M-%S")

    rooms_all = npy_data['rooms']
    room_recepts_all = npy_data['room_receptacles']
    
    # load persona data
    with open(persona_data_path, 'rb') as fh:
        persona_data_dict = pkl.load(fh)

    persona_data = persona_data_dict['personas']
    # print(persona_data[0]['seen_keys'])
    # print(persona_data[0]['seen_key_recepts'])
    # print(persona_data[0]['unseen_keys'])
    # print(persona_data[0]['unseen_key_recepts'])

    # one hot encoding matrices for users and rooms
    user_encoding_matrix = F.one_hot(
        torch.arange(0, int(persona_data_dict['config']['num_users'])))
    room_encoding_matrix = F.one_hot(
        torch.arange(0, len(rooms_all)))

    train_pos_data_dict, train_neg_data_dict = generate_train_data(persona_data)
    test_pos_data_dict, test_neg_data_dict = generate_test_data(persona_data)

    bert_generator = DistillBERTEmbeddingGenerator()

    train_pos_tensor_data = tensor_data_with_bert(train_pos_data_dict, bert_generator)
    train_neg_tensor_data = tensor_data_with_bert(train_neg_data_dict, bert_generator)
    test_pos_tensor_data = tensor_data_with_bert(test_pos_data_dict, bert_generator)
    test_neg_tensor_data = tensor_data_with_bert(test_neg_data_dict, bert_generator)

    train_tensor_data = dict()
    test_tensor_data = dict()

    for tensor_data in [train_pos_tensor_data, train_neg_tensor_data, test_pos_tensor_data, test_neg_tensor_data]:
        for key, value in tensor_data.items():

            assert value['is_train'] in [True, False]

            if value['is_train']:
                train_tensor_data[key] = value

            else:
                test_tensor_data[key] = tensor_data[key]

    torch.save(dict({'config': persona_data_dict['config'],
                    'train': train_tensor_data, 
                    'test': test_tensor_data}), 'personas_tensor_data_{}.pt'.format(timestampStr))


if __name__ == '__main__':

    np.random.seed(8213546)
    torch.manual_seed(8213546)

    assert os.path.exists(sys.argv[1]), 'File not found: {}'.format(sys.argv[1])
    main(sys.argv[1])