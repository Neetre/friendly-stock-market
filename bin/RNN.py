'''
This script is used to train a Recurrent Neural Network (RNN) model to predict stock prices.

Neetre 2024
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


class StockRNN(nn.Module):
    def __init__(self):
        super(StockRNN, self).__init__()
        self.rnn = nn.LSTM(28, 64, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(-1, 28, 28)
        x, hidden =self.rnn(x)

        x = x[:, -1, :]
        x = self.batchnorm(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    

def load_data():
    # Load the preprocessed data
    df = pd.read_csv('../data/csv_preprocessed/NVDA.csv')
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)    
    return df


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    df = load_data()

    # Split the data into training and testing sets
    train_data = df.iloc[:int(0.8*len(df))]
    x = train_data.drop('Close', axis=1)
    y = train_data['Close']
    train_data = (torch.tensor(x.values), torch.tensor(y.values))

    test_data = df.iloc[int(0.8*len(df)):]
    x = test_data.drop('Close', axis=1)
    y = test_data['Close']
    test_data = (torch.tensor(x.values), torch.tensor(y.values))

    # Create the model
    model = StockRNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, 20):
        train(model, device, train_data, optimizer, epoch)
        test(model, device, test_data)
        scheduler.step()


if __name__ == '__main__':
    main()
