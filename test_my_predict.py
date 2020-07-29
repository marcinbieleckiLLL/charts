import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('BTC_EUR_historical_gdax_newer.csv', sep=',')
data = data[['Date', 'Price']]
data = data.iloc[::-1]

#A variable for predicting 'n' days out into the future
prediction_days = 30 #n = 30 days

#Create another column (the target or dependent variable) shifted 'n' units up
data['Prediction'] = data[['Price']].shift(-prediction_days)

#CREATE THE INDEPENDENT DATA SET (X)

print(data)

# Convert the dataframe to a numpy array and drop the prediction column
X = np.array(data.drop(['Prediction'],1))

#Remove the last 'n' rows where 'n' is the prediction_days
X= X[:len(data)-prediction_days]

#CREATE THE DEPENDENT DATA SET (y)
# Convert the dataframe to a numpy array (All of the values including the NaN's) y = np.array(df['Prediction'])
# Get all of the y values except the last 'n' rows
y = np.array(data['Prediction'])
y = y[:-prediction_days]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

prediction_days_array = np.array(data.drop(['Prediction'],1))[-prediction_days:]

from sklearn.svm import SVR
# Create and train the Support Vector Machine
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)#Create the model
svr_rbf.fit(x_train, y_train) #Train the model

svr_rbf_confidence = svr_rbf.score(x_test, y_test)
print("svr_rbf accuracy: ", svr_rbf_confidence)

svm_prediction = svr_rbf.predict(x_test)
svm_prediction = svr_rbf.predict(prediction_days_array)

plt.figure(figsize=(10, 5))
plt.plot(y_test, color='red', label='Real BTC-EUR Price')
plt.plot(svm_prediction, color='blue', label='LSTM BTC-EUR Prediction')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.grid(True, linestyle='-.')
plt.show()