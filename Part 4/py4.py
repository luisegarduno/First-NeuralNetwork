import numpy as np

inputs   = [ [  1.0,     2.0,   3.0,   2.5], 	# inputs shape: (3,4)
             [  2.0,     5.0,  -1.0,   2.0],
             [ -1.5,     2.7,   3.3,  -0.8] ]


weights  = [ [  0.2,     0.8,  -0.5,   1.0],    # weights shape: (3,4)
             [  0.5,   -0.91,  0.26,  -0.5],
             [-0.26,   -0.27,  0.17,  0.87] ]

biases = [2, 3, 0.5] 						


output = np.dot(inputs, np.array(weights).T) + biases
# We recieve an error after trying to do this because this is what we are trying to do:
#               weights					inputs
#        [  1,1	 1,2  1,3  1,4                  [  1,1	1,2  1,3  1,4
#           2,1  2,2  2,3  2,4          *          2,1  2,2  2,3  2,4
#           3,1  3,2  3,2  3,4 ]                   3,1  3,2  3,2  3,4 ]
#
#   (1,1) * (1,1)  +
#   (1,2) * (2,1)  +
#   (1,3) * (3,1)  +
#   (1,4) * error

print(output)
