import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

np.random.seed(0)

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

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples = 100, classes = 3)

dense1 = layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
