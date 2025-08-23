import numpy as np

import numpy as np

def relu(x):
    x = np.array(x)
    return np.maximum(0, x).tolist()

def leaky_relu(x, alpha=0.01):
    x = np.array(x)
    return np.where(x > 0, x, alpha*x).tolist()

def sigmoid(x):
    x = np.array(x)
    return (1 / (1 + np.exp(-x))).tolist()

def tanh(x):
    x = np.array(x)
    return np.tanh(x).tolist()

def gelu(x):
    x = np.array(x)
    return (0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))).tolist()


# Optional: simulate single-layer output
def layer_output(x, weights=None, bias=None, activation_fn=relu):
    x = np.array(x).reshape(1, -1)  # ensures row vector
    if weights is not None:
        weights = np.array(weights)
        x = x @ weights
    if bias is not None:
        x = x + np.array(bias)
    return activation_fn(x.flatten())

