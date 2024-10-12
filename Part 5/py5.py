import numpy as np

np.random.seed(0)

X	= [ [  1.0,		2.0,   3.0,	  2.5], 	# X shape: (3,4)
		[  2.0,		5.0,  -1.0,	  2.0],
		[ -1.5,		2.7,   3.3,  -0.8] ]

'''
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []


# Simple example of ReLU
for i in inputs:
	if i > 0:
		output.append(i)
	elif i <= 0:
		output.append(0)

# Or

# append the maximum of 0 or i
for i in inputs:
	output.append(max(0,i))

print(output)
'''


class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		# We want numpy to create a weights that will be the size o
		# number of inputs times the number of neurons we want

		# The parameters of randn is the shape,
		# in this case, randn(number of columns, number of rows)
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

		# first parameter in np.zeros is the shape
		# Remember, each neuron has a unique bias, so that is how many
		# biases we will initially generate
		self.biases = np.zeros((1, n_neurons))
		# what randn does is just a Gaussian Distribution around 0

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases



class Activation_ReLU:
	def forward(self,inputs):
		self.outputs = np.maximum(0, inputs)


layer1 = Layer_Dense(4,5)
layer1.forward(X)


layer2 = Layer_Dense(5,2)
layer2.forward(layer1.output)
print(layer2.output)
