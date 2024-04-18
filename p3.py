inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outpus = []

'''
for neuron_weights, neuran_b in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuran_b
    layer_outpus.append(neuron_output)
        
print(layer_outpus)
'''

list = [1, 2, 3, 4]
#shape of list: [4,]
#Type: 1D array, vector

lol = [[1, 2, 3, 4],
       [3, 4, 5, 6]] # list of list
#shape of list: [2, 4], 2 list of list, 2 in fisrt dimension and 4 in second dimension
#Type: 2D array, matrix

lol = [[[1, 2, 3, 4],
       [3, 4, 5, 6]],
       [[1, 2, 3, 4],
       [3, 4, 5, 6]],
       [[1, 2, 3, 4],
       [3, 4, 5, 6]]] # list of list of list
#shape of list: [3, 2, 4], 3 in first dimension, 2 in second dimension and 4 in last dimension
#Type: 3D array, matrix


