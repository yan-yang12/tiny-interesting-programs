import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras import optimizers
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# This program uses LSTM to predict the population of two species that follows Lotka-Volterra equations.
# a, b, c, d correspond to the parameters of the equations, and the initial population is given by y0.

# -----------------------------parameters-----------------------------
a = 1.
b = 0.1
c = 1.5
d = 0.75

# time interval
t = np.linspace(0, 60, 2000)

# initial population
y0 = [10, 5]


def dX_dt(X):
    return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])


data = sp.integrate.odeint(dX_dt, y0, t)

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Use two thirds of the data as training data
N = data.shape[0] // 3 * 2
train_data = data[0:N, :]

x_train = TimeseriesGenerator(train_data, train_data, length=8)

# ------------------------LSTM------------------------
model = Sequential()
model.add(LSTM(8, input_shape=(8, 2)))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(2, activation='tanh'))
adam = optimizers.adam(lr=0.001)
model.compile(adam, loss='mse')

model.fit_generator(x_train, epochs=2000)

# ---------------------prediction----------------------
pred = []
batch = train_data[-8:, :].reshape((1, 8, 2))
for i in range(data.shape[0] - N):
    pred.append(model.predict(batch)[0])
    batch = np.append(batch[:, 1:, :], [[pred[i]]], axis=1)

pred = scaler.inverse_transform(pred)
data = scaler.inverse_transform(data)

plt.figure(1)
p1, p2 = plt.plot(t[N:], pred, linestyle='--', alpha=0.9, linewidth=1.5, color='r')
p3, p4 = plt.plot(t[N - 500:], data[N - 500:, :], alpha=0.6, linewidth=2)
plt.legend([(p1, p2), (p3, p4)], ['Predicted', 'Actual'])
plt.title('Predicted and Actual Signals')
plt.xlabel('Time')
plt.ylabel('Population')

plt.figure(2)
test_data = data[N:, :]
l2_err = np.linalg.norm((pred - test_data), ord=2, axis=1)
plt.plot(t[N:], l2_err)
plt.title('L2 Norm Error')
plt.xlabel('Time')
plt.ylabel('L2 Norm Error')
plt.show()
