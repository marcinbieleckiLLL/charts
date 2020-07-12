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

    for i in range(len(seq)):
        end = i + n_steps_in
        out_end = end + n_steps_out

        if out_end > len(seq):
            break

        seq_x, seq_y = seq[i:end], seq[end:out_end]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


prices = get_prices("BTC_EUR_historical_gdax.csv")
futurePrices = prices[len(prices) - 100:len(prices)]
prices = prices[0:len(prices)-100]

# How many periods looking back to train
n_per_in = 30

# How many periods ahead to predict
n_per_out = 1

# Features (in this case it's 1 because there is only one feature: price)
n_features = 1

# Splitting the data into appropriate sequences
X, y = split_sequence(prices, n_per_in, n_per_out)

# Reshaping the X variable from 2D to 3D
X = X.reshape((X.shape[0], X.shape[1], n_features))

# LSTM Model parameters, I chose
batch_size = 2  # Batch size (you may try different values)
epochs = 150  # Epoch (you may try different values)
loss = 'mean_squared_error'  # Since the metric is MSE/RMSE
optimizer = 'rmsprop'  # Recommended optimizer for RNN
activation = 'linear'  # Linear activation
input_shape = (n_per_in, n_features)  # Input dimension
output_dim = n_per_out  # Output dimension

'''model = Sequential()
model.add(LSTM(units=output_dim, return_sequences=True, input_shape=input_shape))
model.add(Dense(units=32, activation=activation))
model.add(LSTM(units=output_dim, return_sequences=False))
model.add(Dense(units=output_dim, activation=activation))
model.compile(optimizer=optimizer, loss=loss)

model.fit(x=X,
          y=y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.05)

model.save('coin_predictor.h5')'''
model = load_model('coin_predictor.h5')


testX, testY = split_sequence(futurePrices, n_per_in, n_per_out)
testX = testX.reshape((testX.shape[0], testX.shape[1], n_features))
testY = np.concatenate(testY).ravel().tolist()

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