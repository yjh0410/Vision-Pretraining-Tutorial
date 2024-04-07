import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    y = sigmoid(x)
    return y * (1 - y)

def relu(x):
    flag = (x >= 0).astype(np.float32)
    x = x * flag

    return x

def relu_grad(x):
    flag = (x >= 0).astype(np.float32)
    return flag

# sigmoid
x = np.arange(-5, 5, 0.2)
y = sigmoid(x)
plt.plot(x, y)

# relue
y = relu(x)
plt.plot(x, y)

plt.show()

# sigmoid grad
x = np.arange(-5, 5, 0.2)
y = sigmoid_grad(x)
plt.plot(x, y)

# relue
y = relu_grad(x)
plt.plot(x, y)

plt.show()