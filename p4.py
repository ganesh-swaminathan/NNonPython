import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 3.7, 3.3, -0.8]]


class layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #n_inputs is the size if the dimension and n_neurons is the size of the network or number of neurons as output
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)

layer1 = layer_Dense(4, 3)
#each layer input will be the output of the previous layer, the size should match
layer2 = layer_Dense(3, 6)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)