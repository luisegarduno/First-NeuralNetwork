import numpy as np

inputs   = [1.0,     2.0,   3.0,   2.5]
weights  = [0.2,     0.8,  -0.5,   1.0]
bias = 2

# At the moment the order inputs & weights are inputted into np dot product does not matter so much
# But later on it will, so it is recommended that 'weights' goes first then 'inputs'
output = np.dot(weights,inputs) + bias

# output = np.dot(weights, inputs) + bias
#	-> np.dot( [0.2, 0.8, -0.5, 1.0], [1.0, 2.0, 3.0, 2.5])
#	-> (weights[0] * inputs[0]) + (weights[1] * inputs[1]) + (weights[2] * inputs[2]) + (weights[3] * inputs[3])
#	-> (0.2 * 1.0) + (0.8 * 2.0) + (-0.5 * 3.0) + (1.0 * 2.5)
# output = 2.8

print("Neuron Output: ", output)

# This file simply shows a tutorial as to how to use numpy to do dot product to solve for the output of 1 neuron