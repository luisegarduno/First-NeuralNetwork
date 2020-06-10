inputs   = [1.0,     2.0,   3.0,   2.5] 	# inputs = x_1,i = [ x_1,1  ,  x_1,2  ,  x_1,3  ,  x_1,4]

weights1 = [0.2,     0.8,  -0.5,   1.0] 	# weights1 = w_1,i,j = [  w_1,1,1  ,  w_1,2,1  ,  w_1,3,1  ,  w_1,4,1  ]
weights2 = [0.5,   -0.91,  0.26,  -0.5] 	# weights2 = w_2,i,j = [  w_2,1,1  ,  w_2,2,1  ,  w_2,3,1  ,  w_2,4,1  ]
weights3 = [-0.26, -0.27,  0.17,  0.87] 	# weights3 = w_3,i,j = [  w_3,1,1  ,  w_3,2,1  ,  w_3,3,1  ,  w_3,4,1  ]

bias1 = 2				   					# bias1 = b_1,1
bias2 = 3				   					# bias2 = b_1,2
bias3 = 0.5				   					# bias3 = b_1,3

output =  [	(inputs[0] * weights1[0]) + (inputs[1] * weights1[1]) + (inputs[2] * weights1[2]) + (inputs[3] * weights1[3]) + bias1,
			(inputs[0] * weights2[0]) + (inputs[1] * weights2[1]) + (inputs[2] * weights2[2]) + (inputs[3] * weights2[3]) + bias2,
			(inputs[0] * weights3[0]) + (inputs[1] * weights3[1]) + (inputs[2] * weights3[2]) + (inputs[3] * weights3[3]) + bias3]

print("Neuron Output: ", output)


# In Part1/py1.py, we looked at 1 single neuron that received 3 inputs,
# In Part2/py2.py, we are looking at an entire layer of neutrons, in which each neutron receives 4 inputs.

# In this tutorial, we are looking at a neural network with the following setup :
# Layer Size:
#				Input Layer = 3
# 			  	Hidden Layer 1 = 4
# 				Hidden Layer 2 = 4
# 				Output Layer = 3

# Each of the neurons in the output layer (3) will recieve an input value from each of the neurons in Hidden Layer 2 (4).
# What's different about this tutorial is that in Part1/py1.py, we only needed 1 set of unique weight values since we were only solving for one neuron.
# But because we are doing an entire layer, this means each unique neuron in the output layer will have a unique weight set,
#	So 3 neurons in output layer means that we will have 3 unique weight sets.
# 	Since there are 4 unique inputs, there will be 4 unique weights. Therefore, each unique weight set will contain 4 values.
#	Also remember that each unique neuron will have its own unique bias value.


# Lastly to get the output of the layer, the process is very similar to the one demostrated in Part1/py1.py.
# For all unique weight sets, do the summation of multipling each input value with the corresponding weight value within the weight set, adding the corresponding unique bias value at the end.