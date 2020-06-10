inputs  = [1.2, 5.1, 2.1] # x = inputs
weights = [3.1, 2.1, 8.7] # w = weights
bias = 3				  # b = bias

# Z or output = Sum (x_1,i * w_1,i,1) + b_i,1
# Z = Sum (x_LayerColumn,Input[]) * (w_ConnectionLevel,Weight,ReceivingNode) + b_LayerColumn,NeuronRow

# Z = (x_1,1 * w_1,1,1) + (x_1,2 * w_1,2,1) + (x_1,3 * w_1,3,1) + b_1,1
output =  (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + bias

print("Neuron Output: ", output)


# So from my understanding thus far, using this example for reference,
	# This Layer only contains 1 Neuron.
	# The Neuron is being fed from 3 other Neurons.
		# 1.) Every Neuron has it's own unique input value (the neuron's output, becomes the inputs for the new neuron)
		# 2.) For each of the unique input values, the Neuron has a corresponding unique weight value as well
		# 3.) Every unique Neuron, has a unique bias. 