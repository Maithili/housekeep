import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader


class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.squeeze(1), y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, is_train=True, device=torch.device('cuda')):

        self.data_dict = data
        self.device = device

        self.train_pos = self.data_dict['train-pos']
        self.train_pos_keys = list(self.train_pos.keys())

        self.train_neg = self.data_dict['train-neg']
        self.train_neg_keys = list(self.train_neg.keys())

        self.test_pos = self.data_dict['test-pos']
        self.test_pos_keys = list(self.test_pos.keys())

        self.test_neg = self.data_dict['test-neg']
        self.test_neg_keys = list(self.test_neg.keys())

        self.is_train = is_train
        self.input_tensor_dim = \
            self.train_pos[self.train_pos_keys[0]]['object_embb'].shape[0] + \
            self.train_pos[self.train_pos_keys[0]]['recept_embb'].shape[0] + \
            self.train_pos[self.train_pos_keys[0]]['room_embb'].shape[0] + \
            self.train_pos[self.train_pos_keys[0]]['user_embb'].shape[0]

    def __getitem__(self, index):

        if self.is_train:

            check_train_pos = [str(index) in k.split('-')[1] for k in self.train_pos_keys]
            check_train_neg = [str(index) in k.split('-')[1] for k in self.train_neg_keys]

            if any(check_train_pos):
                index_match = check_train_pos.index(True)
                return self.train_pos[self.train_pos_keys[index_match]]

            elif any(check_train_neg):
                index_match = check_train_neg.index(True)
                return self.train_neg[self.train_neg_keys[index_match]]

        else:

            check_test_pos = [str(index) in k.split('-')[1] for k in self.test_pos_keys]
            check_test_neg = [str(index) in k.split('-')[1] for k in self.test_neg_keys]

            if any(check_test_pos):
                index_match = check_test_pos.index(True)
                return self.test_pos[self.test_pos_keys[index_match]]

            elif any(check_test_neg):
                index_match = check_test_neg.index(True)
                return self.test_neg[self.test_neg_keys[index_match]]

        raise KeyError(f'Index {index} not found in dataset')

    def __len__(self):

        if self.is_train:
            return len(self.train_pos) + len(self.train_neg)

        else:
            return len(self.test_pos) + len(self.test_neg)

    def return_input_dim(self):
        return self.input_tensor_dim

    def collate_fn(self, data_points):

        # data_points is a list of dicts
        object_embbs = torch.concatenate([torch.tensor(d['object_embb']).unsqueeze(0) for d in data_points], dim=0)
        recept_embbs = torch.concatenate([torch.tensor(d['recept_embb']).unsqueeze(0) for d in data_points], dim=0)
        room_embbs = torch.concatenate([torch.tensor(d['room_embb']).unsqueeze(0) for d in data_points], dim=0)
        user_embbs = torch.concatenate([torch.tensor(d['user_embb']).unsqueeze(0) for d in data_points], dim=0)

        input_tensor = torch.cat([object_embbs, recept_embbs, room_embbs, user_embbs], dim=1)
        labels = torch.tensor([d['ground_truth_score'] for d in data_points])

        # print(input_tensor.shape) #DEBUG
        # print(labels.shape)
        # input('wait')

        return input_tensor.to(self.device), labels.type(torch.float).to(self.device)

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data 
    data_dict = torch.load('./preferences-by-disagreement/personas_tensor_data.pt')
    print(data_dict.keys())

    # create dataset class
    train_dataset = Dataset(data_dict, is_train=True, device=device)
    test_dataset = Dataset(data_dict, is_train=False, device=device)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                                collate_fn=train_dataset.collate_fn)

    # instantiate model
    model = MLP(input_size=train_dataset.return_input_dim(), hidden_size=256, output_size=1)

    trainer = pl.Trainer(devices=1, max_epochs=100)
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()