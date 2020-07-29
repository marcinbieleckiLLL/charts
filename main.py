import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


def get_prices(fileName):
    df = pd.read_csv(fileName)
    prices = np.array(df['Price'])[::-1]

    return scaler.fit_transform(prices.reshape(len(prices), 1))


def split_sequence(seq, n_steps_in, n_steps_out):
    X, y = [], []

    for i in range(len(seq) - n_steps_in):
        a = seq[i:(i + n_steps_in), 0]
        X.append(a)
        y.append(seq[i + n_steps_in, 0])
    return np.array(X), np.array(y)


prices = get_prices("BTC_EUR_historical_gdax.csv")
futurePrices = prices[len(prices) - 58:len(prices)]
prices = prices[0:len(prices)-58]

# How many periods looking back to train
n_per_in = 1

# How many periods ahead to predict
n_per_out = 1

# Features (in this case it's 1 because there is only one feature: price)
n_features = 1

# Splitting the data into appropriate sequences
X, y = split_sequence(prices, n_per_in, n_per_out)

# Reshaping the X variable from 2D to 3D
X = X.reshape((X.shape[0], X.shape[1], n_features))

# LSTM Model parameters, I chose
batch_size = 16  # Batch size (you may try different values)
epochs = 100  # Epoch (you may try different values)
loss = 'mean_squared_error'  # Since the metric is MSE/RMSE
optimizer = 'adam'  # Recommended optimizer for RNN
activation = 'linear'  # Linear activation
input_shape = (n_per_in, n_features)  # Input dimension
output_dim = n_per_out  # Output dimension

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=input_shape))
model.add(LSTM(256))
model.add(Dense(output_dim))
model.compile(optimizer=optimizer, loss=loss)

model.fit(x=X,
          y=y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.05,
          shuffle=False)

model.save('coin_predictor.h5')
model = load_model('coin_predictor.h5')

testX, testY = split_sequence(futurePrices, n_per_in, n_per_out)
testX = testX.reshape((testX.shape[0], testX.shape[1], n_features))
#testY = np.concatenate(testY).ravel().tolist()

preds = model.predict(testX, batch_size=2)
preds = scaler.inverse_transform(preds.reshape(-1, 1))

plt.figure(figsize=(10, 5))
plt.plot(scaler.inverse_transform(np.reshape(testY, (-1, 1))), color='red', label='Real BTC-EUR Price')
plt.plot(preds, color='blue', label='LSTM BTC-EUR Prediction')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.grid(True, linestyle='-.')
plt.savefig("predict")