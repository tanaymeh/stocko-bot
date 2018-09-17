import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("GOOG.csv")

L = len(data)

print(L)

high = np.array([data.iloc[:,2]])
low = np.array([data.iloc[:,3]])
close = np.array([data.iloc[:,4]])

H, = plt.plot(high[0,:])
L, = plt.plot(low[0,:])
C, = plt.plot(close[0,:])

plt.legend([H,L,C],["High","Low","Close"])
plt.show(block=False)

X = np.concatenate([high,low],axis=0)

X = np.transpose(X)
plt.plot(X)

Y = close
Y = np.transpose(Y)
plt.plot(Y)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler1 = MinMaxScaler()
scaler1.fit(Y)
Y = scaler1.transform(Y)

X = np.reshape(X,(X.shape[0],1,X.shape[1]))
print(X.shape)

model = Sequential()
model.add(LSTM(100,activation="tanh",input_shape=(1,2),recurrent_activation="hard_sigmoid"))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer="rmsprop",metrics=[metrics.mae])
newfit = model.fit(X,Y,epochs=15,batch_size=50,verbose=2)

Predict = model.predict(X,verbose=1)
print(Predict)

plt.figure(2)
plt.scatter(Y,Predict)
plt.show(block=False)
plt.figure(3)

Test, = plt.plot(Y)

Predict, = plt.plot(Predict)

plt.legend([Predict, Test],["Predicted Data","Real Data"])
plt.show()