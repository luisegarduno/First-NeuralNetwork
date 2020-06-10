inputs   = [1.0,     2.0,   3.0,   2.5] 	# inputs  = x_1,i = [ x_1,1  ,  x_1,2  ,  x_1,3  ,  x_1,4]

weights = [ [0.2,     0.8,  -0.5,   1.0],	# weights =  w_1,i,j = [  w_1,1,1  ,  w_1,2,1  ,  w_1,3,1  ,  w_1,4,1  ]
			[0.5,   -0.91,  0.26,  -0.5],	#                      [  w_2,1,1  ,  w_2,2,1  ,  w_2,3,1  ,  w_2,4,1  ]
			[-0.26, -0.27,  0.17,  0.87] ]	# 					   [  w_3,1,1  ,  w_3,2,1  ,  w_3,3,1  ,  w_3,4,1  ]

biases = [2, 3, 0.5] 						# biases  = [ b_1,1  ,  b_1,2  ,  b_1,3]


# output value of every neuron of current layer
layer_outputs = []


# Remember that each unique neuron within a layer will have:
#	1.) Unique weight set
#	2.) Unique bias

# iterate through each each set with corresponding weight value
for neuron_weights, neuron_bias in zip(weights, biases):

	# initialize the output value of given neuron to null
	neuron_output = 0

	# iterate through each input value and it's corresponding unique weight value
	for neuron_input, weight in zip(inputs, neuron_weights):

		# Summation of every input times corresponding weight
		neuron_output += neuron_input * weight

	# Add corresponding bias value to current neuron_output
	neuron_output += neuron_bias

	layer_outputs.append(neuron_output)

print("Layer Output: ", layer_outputs)