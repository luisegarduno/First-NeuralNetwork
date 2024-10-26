import numpy as np

np.random.seed(0)

X	= [ [  1.0,     2.0,   3.0,   2.5], 	# X shape: (3,4)
            [  2.0,     5.0,  -1.0,   2.0],
            [ -1.5,     2.7,   3.3,  -0.8] ]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # We want numpy to create a weights that will be the size of
        # number of inputs times the number of neurons we want

        # The parameters of randn is the shape
        # in this case, randn(number of columns, number of rows)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        # first parameter in np.zeros is the shape
        # Remember, each neuron has a unique bias, so that is how many
        # biases we will initially generate
        self.biases = np.zeros((1, n_neurons))
        # what randn does is just a Gaussian Distribution around 0

        def forward(self, inputs):
            self.output = np.dot(inputs, self.weights) + self.biases

# This produces values greater than 1
# print(np.random.randn(4,3))

# This helps us calibrate our weights to make sure it output values under 1
# print(0.10 * np.random.randn(4,3))

# First parameter, how many features in each sample(X), in this case we have 4
# Second parameter,(number of neurons) anything you want
layer1 = Layer_Dense(4, 5)

layer1.forward(X)
#print(layer1.output)


# Only requirement for layer2, is that the first parameter has to be the
# output of layer1, in this situation it is 5
layer2 = Layer_Dense(5, 2)
layer2.forward(layer1.output)
# It output a matrix of size 3, since that was the batch size in the X (inputs),
# with 2 columns since that the number of neurons stated for this layer
print(layer2.output)
