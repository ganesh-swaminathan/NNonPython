import numpy as np
import nnfs
from nnfs.datasets import spiral_data

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 3.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)


class layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #n_inputs is the size if the dimension and n_neurons is the size of the network or number of neurons as output
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = layer_Dense(2, 3)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(activation1.output)
