# Time-Series-Forecasting-with-LSTM-Neural-Network-Python
This project was written in Python using keras and tensorflow. The dataset used in training the LSTM Neural network model was gotten from quandle
# Time series forecasting with LSTM Neural network Python
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
import keras
from keras import Sequential
from keras.layers import Dense, LSTM
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import math


# loading the dataset from quandle
df = quandl.get("FRED/WPU3022")
print(df.head())

# to plot the dataset
plt.plot(df)
plt.xlabel('data')
plt.ylabel('value')
plt.show()

# for the reproducibility of the result
np.random.seed(seed=1234)

df = df.values
df = df.astype(dtype='float32')

# Normalize the dataset
scaler =  MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)
# spliting the dataset into train and test
# train, test = sklearn.model_selection.train_test_split(df, test_size=0.1)
# print(len(train))
train_size = int(len(df)*0.7)
test_size = len(df)-train_size
print(train_size)
print(test_size)

train, test = df[0:train_size,:], df[train_size:len(df),:]
print('\nTrain\n', train)
print('\nTest\n', test)
print(len(train),len(test))

# create a dataset matrix
def create_dataset(df,look_back=1): # look_back is mainly used to predict the future value using the previous value
    dataX, dataY = [],[]
    for i in range (len(df)-look_back-1):
        a = df[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(df[i+look_back,0])
    return np.array(dataX), np.array(dataY)

# reshape the dataset X = current time and Yt+1 = future time period
look_back = 1
trainX,trainY = create_dataset(train,look_back)
testX,testY = create_dataset(test,look_back)
# To do a reshaping of the data
trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))
print('Train data : {} \n Test data : {}'.format(trainX,testX))



# lets create an LSTM  or RNN model
model = keras.Sequential()
model.add(LSTM (4,input_shape = (1,look_back)))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
model.fit(trainX,trainY, batch_size=1,epochs=10,verbose=2)

history = model.evaluate(testX,testY)

# To make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

print('\nTrain predict:\n', train_predict)
print('\nTest predict:\n', test_predict)

# reserve the predicted value to the actual time value using the reserve funtion
train_predict = scaler.inverse_transform(train_predict)
print(train_predict)
test_predict = scaler.inverse_transform(test_predict)
print(test_predict)

trainY = scaler.inverse_transform([trainY])
print(trainY)
testY = scaler.inverse_transform([testY])
print(testY)


# To calculate the root mean square error
test_score = math.sqrt(mean_squared_error(testY[0],test_predict[:,0]))
print('Test score: %.2f RMSE' % test_score)

train_score = math.sqrt(mean_squared_error(trainY[0],train_predict[:,0]))
print('Train  score: %.2f RMSE'% train_score )



