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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def load_data():
    df = pd.read_csv('../data/csv_preprocessed/NVDA.csv')
    df = df.sort_values('Date').reset_index(drop=True)
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
