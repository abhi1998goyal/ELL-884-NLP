import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('GOOGL.csv')

# Extract the 'Close' prices
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Define function to create input sequences and corresponding targets
def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return np.array(X), np.array(Y)

# Define sequence length (number of days to consider)
sequence_length = 10

# Create input sequences and targets
X, Y = create_sequences(scaled_data, sequence_length)

# Split data into training and testing sets
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Define RNN model
class SimpleRNN:
    def __init__(self, input_size, output_size, hidden_size=50):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Weights and biases
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Forward pass
        for i, x in enumerate(inputs):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.last_hs[i + 1] = h

        # Output layer
        y = np.dot(self.Why, h) + self.by
        return y, h

    def backward(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)
        d_Why = np.dot(d_y, self.last_hs[n].T)
        d_by = d_y
        d_h = np.dot(self.Why.T, d_y)
        d_Wxh, d_Whh, d_bh = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.bh)
        d_h_next = np.zeros_like(d_h)

        # Backpropagation through time
        for t in reversed(range(n)):
            dh = (1 - self.last_hs[t + 1] ** 2) * d_h
            d_bh += dh
            d_Wxh += np.dot(dh, self.last_inputs[t].T)
            d_Whh += np.dot(dh, self.last_hs[t].T)
            d_h_next = np.dot(self.Whh.T, dh)

        for dparam in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update weights and biases
        self.Wxh -= learn_rate * d_Wxh
        self.Whh -= learn_rate * d_Whh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by

# Train the model
input_size = 1
output_size = 1
hidden_size = 50
model = SimpleRNN(input_size, output_size, hidden_size)

# Training loop
for epoch in range(200):
    for i in range(len(X_train)):
        x, y = X_train[i], Y_train[i]
        y_pred, _ = model.forward(x)
        loss = np.square(y_pred - y).sum()
        dy = 2 * (y_pred - y)
        model.backward(dy)

# Test the model
predictions = []
for i in range(len(X_test)):
    x, y = X_test[i], Y_test[i]
    y_pred, _ = model.forward(x)
    predictions.append(y_pred)

# Denormalize predictions
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Print predictions
print(predictions)
