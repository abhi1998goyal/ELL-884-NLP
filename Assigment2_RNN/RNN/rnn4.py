import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_sequences(in_,out, seq_length):
    X, Y = [], []
    for i in range(len(in_) - seq_length):
        X.append(in_[i:i+seq_length])
        Y.append(out[i:i+seq_length])
    return np.array(X), np.array(Y)

data = pd.read_csv('GOOGL.csv')

all_features = ['Open', 'High', 'Low', 'Volume','Close']

to_predict=['Close']

features = [feature for feature in all_features if feature != to_predict]

in_ = data[features].values
out = data[to_predict].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_in_ = scaler.fit_transform(in_)
scaled_out_ = scaler.fit_transform(out)


sequence_length = 10
X, Y = create_sequences(scaled_in_,scaled_out_, sequence_length)

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

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
            self.last_inputs = inputs
            self.last_hs = {0: h}
            Y_time=[]

            for i, x in enumerate(inputs):
                z_n = np.dot(self.Wxh,x ).reshape(-1, 1) + np.dot(self.Whh, h) + self.bh
                h_n = np.tanh(z_n)
                self.last_hs[i + 1] = h_n
                self.last_zn[i+1] = z_n
                y = np.dot(self.Why, h_n) + self.by
                Y_time.append(y[0][0])

            return Y_time
        
        def backward(self, diff, learn_rate=2e-2):
            n= len(diff)
            dl_dy = diff[n-1][0]-diff[n-1][1]
            dl_dwhy = np.dot(dl_dy,self.last_hs[n].T)
            d_z1 = (1 - self.last_zn[1] ** 2)
            dh1_dwh = d_z1 * self.last_hs[0] #
            prev_dh_dwh = dh1_dwh
            for i in range(1,n):
                prev_dh_dwh = (1 - self.last_zn[i+1] ** 2) * np.sum(self.last_hs[i] ,np.dot(self.Whh,prev_dh_dwh))

            dhn_dwh = (1 - self.last_zn[n] ** 2) * np.sum(self.last_hs[n-1] ,np.dot(self.Whh,prev_dh_dwh))
            dhn_dwh.reshape(hidden_size,1)
            dl_whh = np.dot(np.dot(dl_dy,self.Why).reshape(hidden_size,1),np.transpose(dhn_dwh))








input_size = X_train.shape[-1]  
output_size = 1 
hidden_size = 50
model = SimpleRNN(input_size, output_size, hidden_size)

for epoch in range(200):
    total_loss = 0
    for i in range(len(X_train)):
        x = X_train[i]
        y= list(Y_train[i])
        y_pred_time = model.forward(x)
        square_losses = [0.5*(actual - predicted) ** 2 for actual, predicted in zip(y_pred_time, y)]
        #loss = np.abs(y_pred_time - y) 
        #total_loss += loss
        total_loss = np.sum(square_losses)
       # dy = np.sign(y_pred - y)
        model.backward(list(zip(y_pred_time, y)))
    print(f'Epoch {epoch+1}, Total Loss: {total_loss}')