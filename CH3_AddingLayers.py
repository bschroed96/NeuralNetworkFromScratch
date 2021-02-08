import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

def add_layer():
    inputs = [[1, 2, 3, 2.5],
              [2., 5., -1., 2],
              [-1.5, 2.7, 3.3, -0.8]]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]

    weights2 = [[0.1, -0.14, 0.5],
                [-0.5, 0.12, -0.33],
                [-0.44, 0.73, -0.13]]
    biases2 = [-1, 2, -0.5]

    # first param decides output shape i.e. 3 outputs since 3 entries in weights
    layer_outputs1 = np.dot(inputs, np.array(weights).T) + biases
    layer_outputs2 = np.dot(layer_outputs1, np.array(weights2).T) + biases2
    return layer_outputs2


#
# Dense Layer Class
#
class Layer_Dense:

    # Layer init
    def __init__(self, n_inputs, n_neurons):
        # random initialization of weight and bias values
        # random.randn generates gaussian distrib around 0, ranging from -1 - +1
        # start with small non zero numbs as to not affect training
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    # Forward Pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

def run_layer():
    # create dataset
    x, y = spiral_data(samples=100, classes=3)

    # create dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2,3)

    # perform a forward pass of our training data through this layer
    dense1.forward(x)

    # output
    print(dense1.output[:5])

if __name__ == '__main__':
    # print(add_layer())
    run_layer()