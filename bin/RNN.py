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
from sklearn.preprocessing import MinMaxScaler

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Initial hidden state of the LSTM
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Initial cell state of the LSTM
        out, _ = self.lstm(x, (h0, c0))  # Forward pass through the LSTM layer
        out = self.fc(out[:, -1, :])  # Forward pass through the fully connected layer
        return out


def load_data():
    df = pd.read_csv('../data/csv_preprocessed/NVDA.csv')
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]  # get the input sequence
        y = data[i + seq_length]  # get the target
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train(model, train_loader, optimizer, loss_f):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}')


def test(model, test_loader, loss_f):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_f(output, target).item()
    test_loss /= len(test_loader)
    print(f"Test set: Average loss: {test_loss:.4f}")


def main():
    df = load_data()

    scaler = MinMaxScaler()  # 
    data_normalized = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']].values)
    
    seq_length = 10
    X, y = create_sequences(data_normalized, seq_length)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, shuffle=True)
    test_loader = DataLoader(test_data, shuffle=False)

    # Hyperparam
    input_size = 5
    hiddend_size = 64
    num_layers = 4
    output_size = 5
    model = StockRNN(input_size, hiddend_size, num_layers, output_size).to(device)
    loss_f = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())
    
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, optimizer, loss_f)
        test(model, test_loader, loss_f)

    

if __name__ == '__main__':
    main()
