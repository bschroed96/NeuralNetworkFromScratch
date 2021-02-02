#
# Chapter 2: Coding our first neurons
# This is code which follows the book Neural Network From Scratch. link: https://nnfs.io/

# neurons are collection of inputs with corresponding weights and a bias.
# i.e. this this neuron has 4 inputs, so 4 weights and a single bias for the neuron.
# A layer is a collection of layers.
# The number of outputs corresponds to the nubmer of weights.
# In this instance, we pass 4 inputs into a single layer which has 3 weights.
# We output a value for each weight (neuron).
def four_inputs_three_layers():
    inputs = [1,2,3,2.5]

    weights1 = [0.2, 0.8, -0.5, 1]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]

    bias1 = 2
    bias2 = 3
    bias3 = 0.5

    outputs = [
        # neuron 1
        inputs[0] * weights1[0] +
        inputs[1] * weights1[1] +
        inputs[2] * weights1[2] +
        inputs[3] * weights1[3] + bias1,

        # neuron 2
        inputs[0] * weights2[0] +
        inputs[1] * weights2[1] +
        inputs[2] * weights2[2] +
        inputs[3] * weights2[3] + bias2,

        # neuron 3
        inputs[0] * weights3[0] +
        inputs[1] * weights3[1] +
        inputs[2] * weights3[2] +
        inputs[3] * weights3[3] + bias3
    ]
    return outputs


# previous method is very inefficient, looping is an easy solution to this.
def looped_layer():
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]
    layer_outputs = []
    for neuron_weight, bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weight):
            # perform maths
            neuron_output += n_input*weight
        # add bias
        neuron_output += bias
        layer_outputs.append(neuron_output)
    return layer_outputs


#
# CH2:
# Tensors, Arrays, and Vectors
#
def dot_prod(v1, v2):
    prod = 0
    for a, b in zip(v1,v2):
        prod += a * b
    return prod


#
# CH: 2
# A single Neuron with NumPy
#
import numpy as np

def single_neuron_numpy():
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

    outputs = np.dot(weights, inputs) + bias
    return outputs

#
# A layer of neurons with NumPy
#

def layer_numpy():
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]

    # first param decides output shape i.e. 3 outputs since 3 entries in weights
    layer_outputs = np.dot(weights, inputs) + biases
    return layer_outputs

#
# A Batch of Data
# Page: 44
#

# our input [1,2,3,2.5] is comprised of 4 data points, features. These features collectively form a
# feature set instance / observation / sample.
# batch data could look like an array of our sample input with shape (8, 4) indicating 8 samples.

batch = [[1,5,6,2],
         [3,2,1,3],
         [5,2,1,2],
         [6,4,8,4],
         [2,8,5,3],
         [1,1,9,4],
         [6,6,0,4],
         [8,7,6,4]]

#
# Matrix Product
# Page: 47
#
# [ 1 2 3 ]   [ a b c ]   [ 1*a + 2*d + 3*g 1*b + 2*e + 3*h 1*c + 2*f + 3*j]
# [ 4 5 6 ] * [ d e f ] = [ 4*1 + 5*d + 6*g 4*b + 5*e + 6*h . . .          ]
# [ 7 8 9 ]   [ g h j ]   [ . . .                                          ]

# size of arrays must match as follows:
# (C1, C2) * (M1, M2) , then C2 == M1
# The resulting size is (M1, C2)

#
# Transposition for the Matrix Product
# Page: 50
#

# [ 1 2 3 ] t  [ 1 4 7 ]
# [ 4 5 6 ] =  [ 2 5 8 ]
# [ 7 8 9 ]    [ 3 6 9 ]



if __name__ == '__main__':
    print(four_inputs_three_layers())
    print(looped_layer())
    print(dot_prod([1,2,3], [2,3,4]))
    print(single_neuron_numpy())
    print(layer_numpy())
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
