import numpy as np

# Load the CSV data
data = np.genfromtxt("your_data.csv", delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5, 6))

# Normalize the data
data_max = np.max(data, axis=0)
data_min = np.min(data, axis=0)
data_normalized = (data - data_min) / (data_max - data_min)

# Function to create sequences of data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, 0])  # Predicting the 'Open' column
    return np.array(X), np.array(y)

# Define the sequence length (number of previous days to consider)
seq_length = 5

# Create sequences of data
X, y = create_sequences(data_normalized, seq_length)

# Split the data into training and testing sets
split = int(0.8 * len(data))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Define the RNN model
class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = np.random.randn(hidden_size, input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size)
        self.Why = np.random.randn(output_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.prev_h = np.zeros((self.hidden_size, 1))
        self.inputs = inputs
        self.hs = {}
        for t in range(len(inputs)):
            self.hs[t] = np.tanh(np.dot(self.Wxh, inputs[t].reshape(-1, 1)) + np.dot(self.Whh, self.prev_h) + self.bh)
            self.prev_h = self.hs[t]
        output = np.dot(self.Why, self.hs[len(inputs) - 1]) + self.by
        return output

    def train(self, inputs, targets, learning_rate):
        loss = 0
        # Forward pass
        output = self.forward(inputs)
        # Compute loss
        loss += 0.5 * np.sum((output - targets) ** 2)
        # Backpropagation through time
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(self.prev_h)
        for t in reversed(range(len(inputs))):
            dy = output - targets
            dWhy += np.dot(dy, self.hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - self.hs[t] ** 2) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, self.inputs[t].reshape(1, -1))
            dWhh += np.dot(dhraw, self.hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # Clip to mitigate exploding gradients
        # Update parameters
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
        return loss

# Train the model
input_size = X_train.shape[2]
hidden_size = 64
output_size = 1
learning_rate = 0.001
rnn = RNN(input_size, hidden_size, output_size)
for epoch in range(500):
    for i in range(len(X_train)):
        inputs, targets = X_train[i], y_train[i]
        loss = rnn.train(inputs, targets.reshape(-1, 1), learning_rate)
    if epoch % 50 == 0:
        print('Epoch:', epoch, 'Loss:', loss)

# Test the model
losses = []
for i in range(len(X_test)):
    inputs, targets = X_test[i], y_test[i]
    output = rnn.forward(inputs)
    loss = 0.5 * np.sum((output - targets.reshape(-1, 1)) ** 2)
    losses.append(loss)
print('Test loss:', np.mean(losses))

# Make predictions
predictions = []
for inputs in X_test:
    output = rnn.forward(inputs)
    predictions.append(output[0, 0])

# Denormalize the predictions and actual values
predictions = predictions * (data_max[0] - data_min[0]) + data_min[0]
actual_values = y_test * (data_max[0] - data_min[0]) + data_min[0]

# Print the predictions and actual values
for i in range(len(predictions)):
    print("Predicted Open: {:.2f}, Actual Open: {:.2f}".format(predictions[i], actual_values[i]))
