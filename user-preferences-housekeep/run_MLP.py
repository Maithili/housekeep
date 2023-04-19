from datetime import datetime
import pickle as pkl

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader

torch.manual_seed(2345678)

class MLP(pl.LightningModule):
    def __init__(self, input_size, config):
        super().__init__()
        self.fc1 = nn.Linear(input_size, config['hidden_size'])
        self.fc2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.fc3 = nn.Linear(config['hidden_size'], config['output_size'])

        self.config = config

        self.loss = nn.BCEWithLogitsLoss()
        self.train_log = []

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.squeeze(1), y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.train_log.append(loss.clone().detach().cpu().numpy())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        return optimizer

    def test_step(self, test_batches, batch_idx):

        self.eval()

        #DEBUG (delete later)
        if len(test_batches) == 2:
            x, y = test_batches
            y_hat = self.forward(x)
            y_pred = torch.sigmoid(y_hat.squeeze(1))
            y_pred = torch.round(y_pred)

            # compute confusion matrix metrics
            tp = torch.sum((y == 1) & (y_pred == 1))
            tn = torch.sum((y == 0) & (y_pred == 0))
            fp = torch.sum((y == 0) & (y_pred == 1))
            fn = torch.sum((y == 1) & (y_pred == 0))
            confusion_matrix = torch.tensor([[tp, fp], [fn, tn]])

            # metrics
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

            return f1

        x, y, user_labels = test_batches
        y_hat = self.forward(x)
        y_pred = torch.sigmoid(y_hat.squeeze(1))
        y_pred = torch.round(y_pred)

        return_output = dict({
            'y': y.cpu().numpy(),
            'y_pred': y_pred.cpu().detach().numpy(),
            'user_labels': user_labels
        })

        # compute confusion matrix metrics
        tp = torch.sum((y == 1) & (y_pred == 1))
        tn = torch.sum((y == 0) & (y_pred == 0))
        fp = torch.sum((y == 0) & (y_pred == 1))
        fn = torch.sum((y == 1) & (y_pred == 0))
        confusion_matrix = torch.tensor([[tp, fp], [fn, tn]])

        # metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        # self.log('test_f1', f1, prog_bar=True, on_step=False, on_epoch=True)
        return confusion_matrix, dict({
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        }), return_output


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, user_conditioned=False, is_train=True, device=torch.device('cuda')):

        self.data_dict = data
        self.device = device

        self.train_pos = self.data_dict['train-pos']
        self.train_pos_keys = list(self.train_pos.keys())
        self.train_pos_indextokey = dict({
            int(k.split('-')[1][1:]): k for k in self.train_pos_keys
            })

        self.train_neg = self.data_dict['train-neg']
        self.train_neg_keys = list(self.train_neg.keys())
        self.train_neg_indextokey = dict({
            int(k.split('-')[1][1:]): k for k in self.train_neg_keys
            })

        self.test_pos = self.data_dict['test-pos']
        self.test_pos_keys = list(self.test_pos.keys())
        self.test_pos_indextokey = dict({
            int(k.split('-')[1][1:]): k for k in self.test_pos_keys
            })

        self.test_neg = self.data_dict['test-neg']
        self.test_neg_keys = list(self.test_neg.keys())
        self.test_neg_indextokey = dict({
            int(k.split('-')[1][1:]): k for k in self.test_neg_keys
            })

        self.is_train = is_train
        self.user_conditioned = user_conditioned

        if self.user_conditioned:
            self.input_tensor_dim = \
                self.train_pos[self.train_pos_keys[0]]['object_embb'].shape[0] + \
                self.train_pos[self.train_pos_keys[0]]['recept_embb'].shape[0] + \
                self.train_pos[self.train_pos_keys[0]]['room_embb'].shape[0] + \
                self.train_pos[self.train_pos_keys[0]]['user_embb'].shape[0]
        else:
            self.input_tensor_dim = \
                self.train_pos[self.train_pos_keys[0]]['object_embb'].shape[0] + \
                self.train_pos[self.train_pos_keys[0]]['recept_embb'].shape[0] + \
                self.train_pos[self.train_pos_keys[0]]['room_embb'].shape[0]

    def __getitem__(self, index):

        if self.is_train:
            # print('train index', index)

            if index in self.train_pos_indextokey.keys():

                key_match = self.train_pos_indextokey[index]
                assert key_match in self.train_pos.keys() and key_match in self.train_pos_keys
                
                return dict({
                    'index': key_match,
                    'data': self.train_pos[key_match]  
                })

            elif index in self.train_neg_indextokey.keys():

                key_match = self.train_neg_indextokey[index]
                assert key_match in self.train_neg.keys() and key_match in self.train_neg_keys

                return dict({
                    'index': key_match,
                    'data': self.train_neg[key_match]  
                })

        else:
            # print('test index', index)

            if index in self.test_pos_indextokey.keys():

                key_match = self.test_pos_indextokey[index]
                assert key_match in self.test_pos.keys() and key_match in self.test_pos_keys

                return dict({
                    'index': key_match,
                    'data': self.test_pos[key_match]  
                })

            elif index in self.test_neg_indextokey.keys():
                
                key_match = self.test_neg_indextokey[index]
                assert key_match in self.test_neg.keys() and key_match in self.test_neg_keys

                return dict({
                    'index': key_match,
                    'data': self.test_neg[key_match]  
                })

        raise KeyError(f'Index {index} not found in dataset')

    def __len__(self):

        if self.is_train:
            return len(self.train_pos) + len(self.train_neg)

        else:
            return len(self.test_pos) + len(self.test_neg)

    def return_input_dim(self):
        return self.input_tensor_dim

    def collate_fn(self, data_points):
        ''' Collate function for dataloader.
            Args:
                data_points: A list of dicts'''

        # data_points is a list of dicts
        object_embbs = torch.concatenate([torch.tensor(d['data']['object_embb']).unsqueeze(0) for d in data_points], dim=0)
        recept_embbs = torch.concatenate([torch.tensor(d['data']['recept_embb']).unsqueeze(0) for d in data_points], dim=0)
        room_embbs = torch.concatenate([torch.tensor(d['data']['room_embb']).unsqueeze(0) for d in data_points], dim=0)

        if self.user_conditioned:

            user_embbs = torch.concatenate([torch.tensor(d['data']['user_embb']).unsqueeze(0) for d in data_points], dim=0)

            input_tensor = torch.cat([object_embbs, recept_embbs, room_embbs, user_embbs], dim=1)

        else:

            input_tensor = torch.cat([object_embbs, recept_embbs, room_embbs], dim=1)

        labels = torch.tensor([d['data']['ground_truth_score'] for d in data_points])

        # print(input_tensor.shape) #DEBUG
        # print(labels.shape)
        # input('wait')

        if self.is_train:
            return input_tensor.to(self.device), \
                labels.type(torch.float).to(self.device)

        else:
            return input_tensor.to(self.device), \
                    labels.type(torch.float).to(self.device), \
                    [d['index'] for d in data_points]


def main():
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%m-%Y_%H-%M-%S")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'hidden_size': 512,
        'output_size': 1,
        'batch_size': 64,
        'max_epochs': 10,
        'lr': 1e-4,
        'num_layers': 2,
        'weight_decay': 1e-6,
        'data_path': 'preferences-by-disagreement/personas_tensor_data_18-04-2023_13-24-46.pt',
        'user_conditioned': True
    }

    # load data
    data_dict = torch.load(config['data_path'])

    # create dataset class
    train_dataset = Dataset(data_dict, user_conditioned=config['user_conditioned'], is_train=True, device=device)
    test_dataset = Dataset(data_dict, user_conditioned=config['user_conditioned'], is_train=False, device=device)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                                collate_fn=train_dataset.collate_fn)


    full_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, 
                                collate_fn=train_dataset.collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, 
                                collate_fn=test_dataset.collate_fn)

    print('data loaders are ready')

    # instantiate model
    model = MLP(input_size=train_dataset.return_input_dim(), config=config)
    print('model is ready')

    # before training
    test_batch = next(iter(test_loader))
    model.to(device)
    confusion_matrix, test_metrics, output = model.test_step(test_batch, 0)

    print('confusion matrix: ', confusion_matrix.cpu().numpy())
    print('f1: ', test_metrics['f1'].cpu().numpy())

    # training
    trainer = pl.Trainer(devices=1, max_epochs=config['max_epochs'])
    trainer.fit(model, train_loader)

    # after training
    model.to(device)
    full_train_batch = next(iter(full_train_loader))
    train_f1 = model.test_step(full_train_batch, 0)
    print('f1 train: ', train_f1.cpu().numpy())

    model.to(device)
    confusion_matrix, test_metrics, output = model.test_step(test_batch, 0)

    print('confusion matrix: ', confusion_matrix.cpu().numpy())
    print('f1: ', test_metrics['f1'].cpu().numpy())

    # save results
    with open('results_{}.pkl'.format(timestampStr), 'wb') as f:
        pkl.dump(dict({
            'config:': config,
            'train_loss': model.train_log,
            'confusion_matrix': confusion_matrix.cpu().numpy(),
            'test_metrics': test_metrics,
            'output': output
        }), f)

if __name__ == '__main__':
    main()
