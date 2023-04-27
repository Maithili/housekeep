from datetime import datetime
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb
from pytorch_lightning.loggers import WandbLogger


from torch.utils.data import DataLoader

seed_everything(2345678)

class MLP(pl.LightningModule):
    def __init__(self, input_size, config):
        super().__init__()

        # 2-layer MLP
        self.fc1 = nn.Linear(input_size, config['hidden_size'])
        self.fc2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.fc2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.fc3 = nn.Linear(config['hidden_size'], 1)

        self.config = config

        self.loss = nn.BCEWithLogitsLoss()
        self.train_log = []

        # test values
        self.rooms_hkp = ['bathroom',
                            'bedroom',
                            'childs_room',
                            'closet',
                            'corridor',
                            'dining_room',
                            'exercise_room',
                            'garage',
                            'home_office',
                            'kitchen',
                            'living_room',
                            'lobby',
                            'pantry_room',
                            'playroom',
                            'storage_room',
                            'television_room',
                            'utility_room']

        self.room_encoding_matrix = F.one_hot(
            torch.arange(0, len(self.rooms_hkp)))


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.squeeze(1), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.train_log.append(loss.clone().detach().cpu().numpy())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        return optimizer

    def validation_step(self, val_batches, batch_idx):

        print('val')

        self.eval()

        x, y = val_batches

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

        self.log('val_prec', precision, on_step=True, on_epoch=True, logger=True)
        self.log('val_recl', recall, on_step=True, on_epoch=True, logger=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, logger=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True, logger=True)

        # self.log('test_f1', f1, prog_bar=True, on_step=False, on_epoch=True)
        return confusion_matrix, dict({
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        })


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, user_conditioned=False, is_train=True, device=torch.device('cuda')):

        self.data_dict = data
        self.device = device

        # self.test_data = self.data_dict['test']
        # self.test_data_indextokey = dict({
        #     i: k for i, k in enumerate(self.test_data.keys())
        #     })

        self.is_train = is_train
        self.user_conditioned = user_conditioned

        self.input_tensor_dim = \
            self.data_dict[0]['csr_item_1'].shape[-1] + \
            self.data_dict[0]['clip_item_1'].shape[-1] + \
            self.data_dict[0]['csr_item_2'].shape[-1] + \
            self.data_dict[0]['clip_item_2'].shape[-1] + \
            self.data_dict[0]['room_embb'].shape[-1]

        if self.user_conditioned:
            self.input_tensor_dim += \
                self.data_dict[0]['persona_embb'].shape[-1]

    def __getitem__(self, index):

        return self.data_dict[index]
    
    def __len__(self):

        return len(self.data_dict)

    def return_input_dim(self):
        return self.input_tensor_dim

    def collate_fn(self, data_points):
        ''' Collate function for dataloader.
            Args:
                data_points: A list of dicts'''

        # data_points is a list of dicts
        csr_embbs_1 = torch.cat([torch.tensor(d['csr_item_1']).unsqueeze(0) for d in data_points], dim=0)
        clip_embbs_1 = torch.cat([torch.tensor(d['clip_item_1']) for d in data_points], dim=0)
        csr_embbs_2 = torch.cat([torch.tensor(d['csr_item_2']).unsqueeze(0) for d in data_points], dim=0)
        clip_embbs_2 = torch.cat([torch.tensor(d['clip_item_2']) for d in data_points], dim=0)

        room_embbs = torch.cat([torch.tensor(d['room_embb']).unsqueeze(0) for d in data_points], dim=0)

        if self.user_conditioned:

            user_embbs = torch.cat([torch.tensor(d['persona_embb']).unsqueeze(0) for d in data_points], dim=0)

            input_tensor = torch.cat([csr_embbs_1, clip_embbs_1, csr_embbs_2, clip_embbs_2, room_embbs, user_embbs], dim=1)

        else:

            input_tensor = torch.cat([csr_embbs_1, clip_embbs_1, csr_embbs_2, clip_embbs_2, room_embbs], dim=1)

        labels = torch.tensor([d['label'] for d in data_points])

        
        return input_tensor.to(self.device), \
            labels.type(torch.float).to(self.device)


def main():
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%m_%H-%M")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'hidden_size': 512,
        'batch_size': 32,
        'max_epochs': 50,
        'lr': 1e-4,
        'num_layers': 3,
        'weight_decay': 1e-6,
        'train_data_path': '/srv/rail-lab/flash5/mpatel377/data/csr_clip_preferences/seen_partial_preferences_50600.pt',
        'val_data_path': '/srv/rail-lab/flash5/kvr6/dev/data/csr_clip_preferences/unseen-val_partial_preferences_51800.pt',
        'test_data_path': None,
        'user_conditioned': False
    }

    print('config: ')
    for k, v in config.items():
        print(f'{k}: {v}')

    wandb_logger = WandbLogger(project='user_specific_preferences',
                                   name='mlp_user{}_{}'.format(config['user_conditioned'], 
                                                               timestampStr),
                                   job_type='train')
    wandb_logger.experiment.config.update(config)

    # load data
    train_data_dict = torch.load(config['train_data_path'])
    val_data_dict = torch.load(config['val_data_path'])

    # create dataset class
    train_dataset = Dataset(train_data_dict, user_conditioned=config['user_conditioned'], is_train=True, device=device)
    val_dataset = Dataset(val_data_dict, user_conditioned=config['user_conditioned'], is_train=False, device=device)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                                collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, 
                                collate_fn=val_dataset.collate_fn)

    print('data loaders are ready')

    # instantiate model
    model = MLP(input_size=train_dataset.return_input_dim(), config=config)
    print('model is ready')

    # defining callbacks
    checkpoint_callback_early_stop = ModelCheckpoint(dirpath='/srv/rail-lab/flash5/kvr6/dev/housekeep_csr/user-preferences-housekeep/ckpts',
                                            filename='mlp_user{}_{}'.format(config['user_conditioned'], 
                                                               timestampStr)+'/model-{epoch}-{val_f1:.2f}',
                                            verbose=True, 
                                            monitor='val_f1',  
                                            mode='max',
                                            every_n_val_epochs=1)
    checkpoint_callback = ModelCheckpoint(dirpath='/srv/rail-lab/flash5/kvr6/dev/housekeep_csr/user-preferences-housekeep/ckpts',
                                            filename='mlp_user{}_{}'.format(config['user_conditioned'], 
                                                               timestampStr)+'/model-{epoch}-{val_f1:.2f}',
                                            verbose=True, 
                                            mode='max',
                                            every_n_val_epochs=1)

    learning_rate_callback = LearningRateMonitor(logging_interval='epoch')


    # training loop
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                        check_val_every_n_epoch=1,
                        logger=wandb_logger,
                        callbacks=[learning_rate_callback,
                                checkpoint_callback,
                                checkpoint_callback_early_stop
                                ],
                        checkpoint_callback=True,
                        num_sanity_val_steps=1)

    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model on the held out test set
    trainer.test() #TODO

    # Close wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
