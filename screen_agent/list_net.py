import numpy as np
from torch import nn
import torch
import torch.optim as optim
from logger import logger
import torch.nn.functional as F


device = 'cuda:0'

def ranking_model_training(train_data, train_label, test_data, test_label, hidden_size, output_size, batch_size, epochs):
    training_data_size = len(train_data)
    input_size = len(train_data[0])
    ranking_model = Net(input_size, hidden_size, output_size).to(device)
    opt = optim.Adam(ranking_model.parameters(), lr=10**-8)
    for epoch in range(epochs):
        idx = torch.randperm(training_data_size)
        train_data_shuffle = train_data[idx]
        train_label_shuffle = train_label[idx]

        cur_batch = 0
        average_loss = []
        ranking_model.train()
        for it in range(training_data_size // batch_size):
            batch_train_data = train_data_shuffle[cur_batch: cur_batch + batch_size]
            batch_train_label = train_label_shuffle[cur_batch: cur_batch + batch_size]

            batch_train_data = torch.Tensor(batch_train_data).to(device)
            batch_train_label = torch.Tensor(batch_train_label).to(device)
            cur_batch += batch_size

            opt.zero_grad()
            if len(batch_train_data) > 0:
                batch_pred = ranking_model(batch_train_data)
                batch_loss = list_net_loss(batch_pred, batch_train_label)
                batch_loss.backward()
                opt.step()
                average_loss.append(batch_loss.detach().to('cpu').numpy())
        average_loss = np.average(average_loss)
        logger.info('epoch: {}, train average loss: {}'.format(epoch, average_loss))

        ranking_model.eval()
        with torch.no_grad():
            for it in range(len(test_data) // batch_size):
                batch_test_data = torch.Tensor(test_data[cur_batch: cur_batch + batch_size]).to(device)
                batch_test_label = torch.Tensor(test_label[cur_batch: cur_batch + batch_size]).to(device)
                cur_batch += batch_size

                opt.zero_grad()
                if len(batch_test_data) > 0:
                    batch_pred = ranking_model(batch_test_data)
                    batch_loss = list_net_loss(batch_pred, batch_test_label)
                    batch_loss.backward()
                    average_loss.append(batch_loss.detach().to('cpu').numpy())
                    opt.step()
        average_loss = np.average(average_loss)
        logger.info('epoch: {}, test average loss: {}'.format(epoch, average_loss))
    return ranking_model

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x


def list_net_loss(y_pred, y_true, eps=10**-6):
    """
    https://github.com/allegro/allRank/blob/master/allrank/models/losses/listNet.py
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)
    value = true_smax * preds_log
    loss = torch.mean(-torch.sum(value, dim=1))

    preds_max_numpy = preds_smax.detach().cpu().numpy()
    true_smax_numpy = true_smax.detach().cpu().numpy()
    value_numpy = value.detach().cpu().numpy()
    loss_sum = torch.sum(value, dim=1).detach().cpu().numpy()
    return loss
