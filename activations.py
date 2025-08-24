import numpy as np

# -------------------------
# Activation functions
# -------------------------
def relu(x):
    """ReLU activation"""
    x = np.array(x)
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation"""
    x = np.array(x)
    return np.where(x > 0, x, alpha * x)

def sigmoid(x):
    """Sigmoid activation"""
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh activation"""
    x = np.array(x)
    return np.tanh(x)

def gelu(x):
    """GELU activation"""
    x = np.array(x)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# -------------------------
# Layer output
# -------------------------
def layer_output(x, weights=None, bias=None, activation_fn=relu):
    """
    Computes single-layer output: x @ weights + bias, then activation.
    
    Parameters:
        x : array-like, shape (n_features,) or (n_samples, n_features)
        weights : array-like, shape (input_dim, output_dim)
        bias : array-like, shape (output_dim,)
        activation_fn : callable activation function
    
    Returns:
        np.ndarray of activated outputs
    """
    x = np.array(x)
    
    # If no weights/bias, treat as element-wise activation (for plotting)
    if weights is None and bias is None:
        if activation_fn is not None:
            return activation_fn(x)
        return x
    
    # Neural network layer computation
    x = np.atleast_2d(x)  # ensure 2D: (n_samples, n_features)
    
    if weights is not None:
        weights = np.array(weights)
        if x.shape[1] != weights.shape[0]:
            raise ValueError(f"Weight shape {weights.shape} incompatible with input shape {x.shape}")
        x = x @ weights
    
    if bias is not None:
        bias = np.array(bias).reshape(1, -1)  # ensure row vector for broadcasting
        x = x + bias
    
    if activation_fn is not None:
        x = activation_fn(x)
    
    return x