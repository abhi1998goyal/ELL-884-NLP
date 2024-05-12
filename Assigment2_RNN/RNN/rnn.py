import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def random_sequence_with_sum(target_sum, max_value):
    sequence = []
    current_sum = 0
    
    while current_sum < target_sum:
        rand_num = np.random.randint(3, max_value)
        sequence.append(rand_num)
        current_sum += rand_num
    sequence[-1] -= current_sum - target_sum
    
    return sequence


def create_sequences(in_,out, seq_length,seq):
    X, Y = [], []
    sum=0
    for i in range(len(seq)):
        X.append(in_[sum:sum+seq[i]-1])
        Y.append(out[sum+seq[i]-1])
        sum+=seq[i]
    return X, Y

data = pd.read_csv('GOOGL.csv')

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Calculate the number of days since the start date
start_date = data['Date'].min()
data['Days'] = (data['Date'] - start_date).dt.days



all_features = ['Open', 'High', 'Low', 'Volume','Close','Days']

to_predict=['Close']

features = [feature for feature in all_features if feature != to_predict]

in_ = data[features].values
out = data[to_predict].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaled_in_ = scaler.fit_transform(in_)
scaled_out_ = scaler2.fit_transform(out)


sequence_length = 20

sequence1 = random_sequence_with_sum(data.shape[0], 50)
#X, Y = create_sequences(scaled_in_,scaled_out_, sequence_length,sequence)

split_index = int(len(in_) * 0.93)
X_train, X_test = in_[:split_index], in_[split_index:]
Y_train, Y_test = out[:split_index], out[split_index:]

X_train,Y_train = create_sequences(X_train,Y_train, sequence_length,sequence1)



combined_data = list(zip(X_train, Y_train))
np.random.shuffle(combined_data)
X_train_shuffled, Y_train_shuffled = zip(*combined_data)
X_train = X_train_shuffled
Y_train = Y_train_shuffled

def clip_mat(mat):
    max_norm = 10
    min_norm = -10
    norm = np.linalg.norm(mat)
    if norm > max_norm or norm < min_norm:
        scaling_factor = max_norm / norm
        mat*=scaling_factor
    return mat

class SimpleRNN:
    def __init__(self, input_size, output_size, hidden_size=50):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.last_hs = {0: h}
        Y_time=[]
        s=len(inputs)
        if(s>20):
          inputs=inputs[s-20:s]
        elif(s<20):
          inputs= np.concatenate([np.zeros((20-s, inputs.shape[1])), inputs], axis=0)
        
        self.last_inputs = inputs

        for i, x in enumerate(inputs):
            h_n = np.tanh(np.dot(self.Wxh,x ).reshape(-1, 1) + np.dot(self.Whh, h) + self.bh)
            self.last_hs[i + 1] = h
        y = np.dot(self.Why, h_n) + self.by
           # Y_time.append(y[0][0])

        #return Y_time
        return y
    
    def backward(self, d_y, learn_rate=2e-2):
           # d_y = diff[2][0]-diff[2][1]
            d_y =d_y
            n = len(self.last_inputs)
            d_Why = np.dot(d_y, self.last_hs[n-1].T)
            d_by = d_y
            d_h = np.dot(self.Why.T, d_y).reshape(-1,1)
            d_Wxh, d_Whh, d_bh = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.bh)

            for t in reversed(range(n-1)):
                a= (1 - self.last_hs[t + 1] ** 2)
                dh = a * d_h
                d_bh += np.sum(dh, axis=1, keepdims=True)  
                d_Wxh += np.dot(dh, self.last_inputs[t].reshape(-1, 1).T)
                d_Whh += np.dot(dh, self.last_hs[t].T)
                d_h = np.dot(self.Whh.T, dh)


            self.Wxh -= learn_rate * clip_mat(d_Wxh)
            self.Whh -= learn_rate * clip_mat(d_Whh)
            self.Why -= learn_rate * clip_mat(d_Why)
            self.bh -= learn_rate * clip_mat(d_bh)
            self.by -= learn_rate * clip_mat(d_by)


input_size = 6
#X_train.shape[-1] 
output_size = 1  
hidden_size = 100
model = SimpleRNN(input_size, output_size, hidden_size)

for epoch in range(2000):
    total_loss = 0
    for i in range(len(X_train)):
        x = X_train[i]
        y= Y_train[i]
        y_pred_time = model.forward(x)
        #square_losses = [0.5*(actual - predicted) ** 2 for actual, predicted in zip(y_pred_time, y)]
        loss = y_pred_time - y
        total_loss += np.abs(loss)
        #total_loss = np.sum(square_losses)
       # dy = np.sign(y_pred - y)  # Gradient for absolute loss function
        model.backward(loss)
    print(f'Epoch {epoch+1}, Total Loss: {total_loss}')


predictions = []
original=[]
for i in range(len(X_test)):
    x, y = X_test[i], Y_test[i]
    y_pred = model.forward(x)
    y_pred_original_scale = scaler2.inverse_transform(np.array(y_pred).reshape(-1, 1))
    y_original_scale = scaler2.inverse_transform(np.array(y).reshape(-1, 1))
    predictions.append(y_pred_original_scale[0][0])
    original.append(y_original_scale[0][0])

#predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

predictions = np.array(predictions)
#print(predictions)

import matplotlib.pyplot as plt

plt.plot(original, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()
