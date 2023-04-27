import os
import sys
import torch
from torch.nn import functional as F

sys.path.append('./CSR')

from user_preferences_housekeep.run_pref_trainer import MLP, Dataset
from CSR.dataloaders.preference_dataset import PreferenceDataset
from CSR.dataloaders.contrastive_dataset import object_key_filter, rooms_hkp
from CSR.shared.data_split import DataSplit

ROOT = '/coc/'

GROUND_TRUTH_FILE = '/coc/flash5/mpatel377/data/csr/scene_graph_edges_complete.pt'
GROUND_TRUTH_NAMES_FILE = '/coc/flash5/mpatel377/data/csr/scene_graph_edges_names.pt'
if not os.path.exists(GROUND_TRUTH_FILE): 
    GROUND_TRUTH_FILE.replace('coc','srv/rail-lab')
    GROUND_TRUTH_NAMES_FILE.replace('coc','srv/rail-lab')
    ROOT = '/srv/rail-lab/'

config = {
    'hidden_size': 512,
    'batch_size': 32,
    'max_epochs': 50,
    'lr': 1e-4,
    'num_layers': 3,
    'weight_decay': 1e-6,
    'user_conditioned': False,
    'user_personas_path': ROOT + 'flash5/kvr6/dev/all_preferences_26-04-2023_11-48-18.pt'
}

config['input_dim'] = 2075 if config['user_conditioned'] else 2065

def find_best_ckpt_csr():

    #TODO: test
    csr_ckpt_dir = ROOT + 'flash5/mpatel377/repos/housekeep/CSR/checkpoints/model' 

    best_ckpt = [f for f in os.listdir(csr_ckpt_dir) if f.endswith('.ckpt')]
    best_ckpt.sort(key=lambda x: float(x.split('.')[0].split('-')[-1].split('=')[1]))
    best_ckpt = best_ckpt[0]

    return os.path.join(csr_ckpt_dir, best_ckpt)


def obj_key_to_class(any_key):
    if '.urdf' in any_key: # receptacle: storage_room_0-table_14_0.urdf
        room_recep_compound = any_key.split('.')[0]

        room_indexed, recep_indexed = room_recep_compound.split('-')

        room_name_split = [k for k in room_indexed.split('_') if not k.isdigit()] # [storage, room]
        recep_name_split = [k for k in recep_indexed.split('_') if not k.isdigit()] # [table]

        final_name = '_'.join(room_name_split) + '|' + '_'.join(recep_name_split)
        return final_name

    else: # object: condiment_1

        any_name_split = [k for k in any_key.split('_') if not k.isdigit()]

        return '_'.join(any_name_split)


def return_gt_from_persona(obj, persona_id):
    ''' Returns all ground truth receptacles for a given object and persona id '''

    obj_class = obj_key_to_class(obj)
    all_gts = []

    for room in rooms_hkp:

        if f'{obj_class}/{room}' in user_persona_dict[f'persona_{persona_id}']['seen']:
            assert _train_obj(object_key_filter.index(obj)), f'Object {obj} is not a training object, {object_key_filter.index(obj)}'
            all_gts += user_persona_dict[f'persona_{persona_id}']['seen'][f'{obj_class}/{room}']

        else:

            if f'{obj_class}/{room}' in user_persona_dict[f'persona_{persona_id}']['unseen-val']:
                assert not _train_obj(object_key_filter.index(obj)), f'Object {obj} is a training object, {object_key_filter.index(obj)}'
                all_gts += user_persona_dict[f'persona_{persona_id}']['unseen-val'][f'{obj_class}/{room}']

            elif f'{obj_class}/{room}' in user_persona_dict[f'persona_{persona_id}']['unseen-test']:
                assert not _train_obj(object_key_filter.index(obj)), f'Object {obj} is a training object, {object_key_filter.index(obj)}'
                all_gts += user_persona_dict[f'persona_{persona_id}']['unseen-test'][f'{obj_class}/{room}']

            # else: # TODO: skip
            #     print(f'Object {obj_class}/{room} not found in persona {persona_id}')
            #     continue

    return all_gts


# ranking model checkpoint
if config['user_conditioned']:
    best_ckpt_pref_ranking = '/srv/rail-lab/flash5/kvr6/dev/housekeep_csr/user_preferences_housekeep/ckpts/mlp_userTrue_27-04_12-42/model-epoch=4-val_f1=0.75-v1.ckpt'
else:
    best_ckpt_pref_ranking = '/srv/rail-lab/flash5/kvr6/dev/housekeep_csr/user_preferences_housekeep/ckpts/mlp_userFalse_27-04_12-46/model-epoch=4-val_f1=0.77-v1.ckpt'

# load persona dictionary: map of persona to object to receptacle
user_persona_dict = torch.load(config['user_personas_path'])

# instantiate model
model = MLP.load_from_checkpoint(best_ckpt_pref_ranking, 
                                 input_size=config['input_dim'],
                                 config=config)
model.eval()

# indexed data: list of tuples (obj_id, epstep rec, receptacle id, epstep rec, room_name, label [GLOBAL])
test_data = PreferenceDataset(ROOT+'flash5/kvr6/dev/data/csr_full_v2_test_26-04-2023_13-51-28', 
                              data_split=DataSplit.TEST, 
                              test_unseen_objects=False, 
                              csr_ckpt_path=find_best_ckpt_csr(),
                              input_room_embedding = True)

obj_names= torch.load(GROUND_TRUTH_NAMES_FILE)['objects']
assert (obj_names == object_key_filter), "Object names in ground truth file do not match with object names in code."

_train_obj = lambda o: (0 <= o < 90) or o > 105 #TODO: make this consistent!!
average_accuracy = {'seen':[[] for i in range(10)],'unseen':[[] for i in range(10)],'total':[[] for i in range(10)]}
total_correct = {'seen':0,'unseen':0,'total':0}
total_total = {'seen':0,'unseen':0,'total':0}

user_encoding_matrix = F.one_hot(torch.arange(0, 10))

while True:
    # gives the mean csr-clip-room input vector per obj-recep for one episode
    print('Episode: ', test_data.episode_curr)
    data = test_data.get_next_episode()

    if data is None: break # end of data

    for persona_id in range(10): 

        # Evaluating episode {test_data.episode_curr} and persona {persona_id}

        ep_bypersona_corr = {'seen':0,'unseen':0,'total':0}
        ep_bypersona_total = {'seen':0,'unseen':0,'total':0}

        for obj_id in data: # data is dictionary with structure: obj - rec - (input, GLOBAL label)
            pred = None
            pred_rank = -float('inf')
            gt = return_gt_from_persona(object_key_filter[obj_id], 
                                        persona_id)     
            for rec_id in data[obj_id]:
                input_vec = data[obj_id][rec_id][0]
                assert len(input_vec.size()) == 2, f'Input vector is not 2D: {input_vec.size()}'


                if config['user_conditioned']:
                    input_vec = torch.cat([input_vec, 
                                    user_encoding_matrix[persona_id].unsqueeze(0)], dim=1)
                    assert input_vec.size() == (1, 2075), f'Input vector is not of size 2075: {input_vec.size()}'
                else:
                    assert input_vec.size() == (1, 2065), f'Input vector is not of size 2065: {input_vec.size()}'

                rank = model(input_vec)

                if rank > pred_rank:
                    pred = rec_id
                    pred_rank = rank

            pred = obj_key_to_class(object_key_filter[pred])
            corr = 1 if pred in gt else 0
            # print(pred, gt, corr)

            # micro accuracy is summed across all personas and episodes and averaged over total number of predictions
            if _train_obj(obj_id):
                total_total['seen'] += 1
                total_correct['seen'] += corr
                ep_bypersona_total['seen'] += 1
                ep_bypersona_corr['seen'] += corr
            else:
                total_total['unseen'] += 1
                total_correct['unseen'] += corr
                ep_bypersona_total['unseen'] += 1
                ep_bypersona_corr['unseen'] += corr

            total_total['total'] += 1
            total_correct['total'] += corr
            ep_bypersona_total['total'] += 1
            ep_bypersona_corr['total'] += corr

        print(f'Persona {persona_id}')
        print(ep_bypersona_corr['seen'], ep_bypersona_total['seen'], total_correct['seen'], total_total['seen'])
        print(ep_bypersona_corr['unseen'], ep_bypersona_total['unseen'], total_correct['unseen'], total_total['unseen'])

        # macro accuracy is summed across all personas and averaged for each episode
        average_accuracy['seen'][persona_id].append(float(ep_bypersona_corr['seen']/ep_bypersona_total['seen']) if ep_bypersona_total['seen'] > 0 else 0)
        average_accuracy['unseen'][persona_id].append(float(ep_bypersona_corr['unseen']/ep_bypersona_total['unseen']) if ep_bypersona_total['unseen'] > 0 else 0)
        average_accuracy['total'][persona_id].append(float(ep_bypersona_corr['total']/ep_bypersona_total['total']) if ep_bypersona_total['total'] > 0 else 0)

    
print("\n\n")

print(f'User conditioning is {config["user_conditioned"]}')

print("~~~~~~~~~~~~~~~~~~~~~FINAL RESULTS~~~~~~~~~~~~~~~~~~~~~")
print("AVERGE ACCURACY (MACRO AVG.) : ",[sum(average_accuracy['total'][pid])/len(average_accuracy['total'][pid]) for pid in range(10)])
print("AVERGE ACCURACY (MICRO AVG.) : ",float(total_correct['total']/total_total['total']), f"({total_correct['total']}/{total_total['total']})")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
print("\n\n")
print("~~~~~~~~~~~~~~~~~~~~~SEEN RESULTS~~~~~~~~~~~~~~~~~~~~~")
print("AVERGE ACCURACY (MACRO AVG.) : ",[sum(average_accuracy['seen'][pid])/len(average_accuracy['seen'][pid]) for pid in range(10)])
print("AVERGE ACCURACY (MICRO AVG.) : ",float(total_correct['seen']/total_total['seen']), f"({total_correct['seen']}/{total_total['seen']})")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
print("\n\n")
print("~~~~~~~~~~~~~~~~~~~~~UNSEEN RESULTS~~~~~~~~~~~~~~~~~~~~~")
print("AVERGE ACCURACY (MACRO AVG.) : ",[sum(average_accuracy['unseen'][pid])/len(average_accuracy['unseen'][pid]) for pid in range(10)])
print("AVERGE ACCURACY (MICRO AVG.) : ",float(total_correct['unseen']/total_total['unseen']), f"({total_correct['unseen']}/{total_total['unseen']})")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\n\n")
