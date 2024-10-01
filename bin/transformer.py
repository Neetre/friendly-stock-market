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

# Prepare data for Transformer
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Define the Transformer model
class StockTransformerEncoder(nn.Module):
    def __init__(self, feature_size, d_model, nhead, num_layers, dim_feedforward):
        super(StockTransformerEncoder, self).__init__()
        self.embedding = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
    
    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

# Main function
def main():
    # Parameters
    ticker = "AAPL"
    start_date = "1980-12-12"
    end_date = "2023-01-01"
    n_steps = 60  # Number of time steps to look back
    test_size = 0.2
    d_model = 64
    nhead = 4
    num_layers = 3
    dim_feedforward = 256
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get and preprocess data
    data = get_stock_data(ticker, start_date, end_date)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = prepare_data(data_scaled, n_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).transpose(0, 1).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).transpose(0, 1).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    # print(f"Shape of X_train: {X_train.shape}")
    # print(f"Shape of y_train: {y_train.shape}")
    # print(f"Shape of X_test: {X_test.shape}")
    # print(f"Shape of y_test: {y_test.shape}")

    # Create model, loss function, and optimizer
    model = StockTransformerEncoder(feature_size=1, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, X_train.size(1), batch_size):
            batch_X = X_train[:, i:i+batch_size, :]
            batch_y = y_train[i:i+batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs[-1, :, 0], batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.save("./model/transformer.pth")

    # Make predictions
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)[-1, :, 0].cpu().numpy()

    # Inverse transform predictions
    test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

    # Calculate RMSE
    rmse = np.sqrt(np.mean((test_predictions - y_test_actual)**2))
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Predict future prices
    last_sequence = X_test[:, -1, :].unsqueeze(1)  # Shape: [n_steps, 1, 1]
    # print(f"Shape of last_sequence: {last_sequence.shape}")
    future_predictions = []

    for _ in range(30):  # Predict next 30 days
        with torch.no_grad():
            pred = model(last_sequence)[-1, :, 0]
        # print(f"Shape of pred: {pred.shape}")
        future_predictions.append(pred.item())
        # Update last_sequence for the next prediction
        last_sequence = torch.cat((last_sequence[1:], pred.unsqueeze(0).unsqueeze(1)), dim=0)
        # print(f"Shape of updated last_sequence: {last_sequence.shape}")

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    print("Future 30-day predictions:", future_predictions.flatten())


if __name__ == "__main__":
    main()
