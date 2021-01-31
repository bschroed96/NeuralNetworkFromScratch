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


if __name__ == '__main__':
    print(four_inputs_three_layers())
    print(looped_layer())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
