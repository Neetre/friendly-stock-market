import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Download stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1)

# Prepare data for RNN
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# Define the RNN model
class StockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
        
    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

# Main function
def main():
    # Parameters
    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = "2024-01-01"
    n_steps = 60  # Number of time steps to look back
    test_size = 0.2
    hidden_size = 50
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32

    # Get and preprocess data
    data = get_stock_data(ticker, start_date, end_date)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = prepare_data(data_scaled, n_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # print(f"Shape of X_train: {X_train.shape}")
    # print(f"Shape of y_train: {y_train.shape}")
    # print(f"Shape of X_test: {X_test.shape}")
    # print(f"Shape of y_test: {y_test.shape}")

    # Create model, loss function, and optimizer
    model = StockRNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.save("./model/stock_rnn.pth")

    # Make predictions
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test).numpy()

    # Inverse transform predictions
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_actual = scaler.inverse_transform(y_test.numpy())

    # Calculate RMSE
    rmse = np.sqrt(np.mean((test_predictions - y_test_actual)**2))
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Predict future prices
    last_sequence = X_test[-1].unsqueeze(0)  # Shape: [1, n_steps, 1]
    # print(f"Shape of last_sequence: {last_sequence.shape}")
    future_predictions = []

    for _ in range(30):  # Predict next 30 days
        with torch.no_grad():
            pred = model(last_sequence)
        # print(f"Shape of pred: {pred.shape}")
        future_predictions.append(pred.item())
        # Update last_sequence for the next prediction
        last_sequence = torch.cat((last_sequence[:, 1:, :], pred.unsqueeze(1)), dim=1)
        # print(f"Shape of updated last_sequence: {last_sequence.shape}")

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    print("Future 30-day predictions:", future_predictions.flatten())

if __name__ == "__main__":
    main()
