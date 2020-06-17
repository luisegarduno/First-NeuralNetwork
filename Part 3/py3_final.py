import numpy as np

inputs   =  [1.0,     2.0,   3.0,   2.5] 	# inputs  = x_1,i = [ x_1,1  ,  x_1,2  ,  x_1,3  ,  x_1,4]

weights = [ [0.2,     0.8,  -0.5,   1.0],	# weights =  w_1,i,j = [  w_1,1,1  ,  w_1,2,1  ,  w_1,3,1  ,  w_1,4,1  ]
			[0.5,   -0.91,  0.26,  -0.5],	#                      [  w_2,1,1  ,  w_2,2,1  ,  w_2,3,1  ,  w_2,4,1  ]
			[-0.26, -0.27,  0.17,  0.87] ]	# 					   [  w_3,1,1  ,  w_3,2,1  ,  w_3,3,1  ,  w_3,4,1  ]

biases = [2, 3, 0.5] 						# biases  =  b_1,j   = [ b_1,1  ,  b_1,2  ,  b_1,3]

output = np.dot(weights, inputs) + biases

print("Layer Output: ", output)


# inputs is a vector
# weights is a matrix containing vector

# The first element you pass is how the return is going to be indexed.
# In this case we want the outputs of each of the neurons.
#	Reading in "weights" first, indexes that there are 3 lists in that list,
#		therefore numpy knows that we want 3 values in our list.
# We want things things indexed by the weight sets. 

# output = np.dot(weights, inputs) + biases
# 	-> np.dot(weights, inputs) = [np.dot(weights[0], inputs), np.dot(weights[1], inputs), np.dot(weights[2], inputs) ]
# 	-> output = [2.8, -1.79, 1.885]
#	-> output = np.dot(weights, inputs) + biases