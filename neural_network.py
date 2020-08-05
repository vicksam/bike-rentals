import numpy as np


class NeuralNetwork(object):
    ''' Simple neural network for a regression problem.
        It uses one hidden layer with a sigmoid activation function.
        Output layer returns its input (f(x) = x activation function).
    '''
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(
            0.0,
            self.input_nodes**-0.5,
            (self.input_nodes, self.hidden_nodes)
        )
        self.weights_hidden_to_output = np.random.normal(
            0.0,
            self.hidden_nodes**-0.5,
            (self.hidden_nodes, self.output_nodes)
        )

        self.lr = learning_rate

        # Define a sigmoid activation function
        # Can do it smartly using lambda
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # Forward pass
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            # Backprop pass
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs,
                hidden_outputs,
                X,
                y,
                delta_weights_i_h,
                delta_weights_h_o
            )

        self.update_weights(
            delta_weights_i_h,
            delta_weights_h_o,
            n_records
        )

    def forward_pass_train(self, X):
        ''' Forward pass

            Arguments
            ---------
            X: features batch

        '''

        ## Hidden layer
        # Signals into hidden layer
        hidden_inputs = X.dot(self.weights_input_to_hidden)
        # Signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        ## Output layer
        # Signals into final output layer
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output)
        # Signals from final output layer
        final_outputs = final_inputs

        # It is a regression problem, so there is no need to flatten the final result
        # into [0. 1] by using sigmoid function
        # In other words using f(x) = x instead of f(x) = sigmoid(x) for activation

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Backward pass

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''

        # Output error
        # Output layer error is the difference between desired target and actual output
        error = y - final_outputs

        # To calculate the amount by which error needs to change in a layer,
        # you need to multiply it by derivative of activation function
        # The result is error term (or delta)
        # error_term = error * f'(x)
        # x is the output from that layer

        # The amount by which the error needs to change
        # Since activation function is f(x) = x the derivative is just f'(x) = 1
        output_error_term = error * 1

        # To calculate the error in the layer you take the error term
        # (amount by which the error needs to change) of the n+1 layer
        # and multiply it by n-th layer's output weights (weights between these two layers)

        # Calculate the hidden layer's contribution to the error
        hidden_error = output_error_term.dot(self.weights_hidden_to_output.T)

        # Layer's error term
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        # Delta weights is the amount with which the weights are to be updated
        # You update the weights by x * error term
        # x is the input to the layer

        # Weight step (input to hidden)
        delta_weights_i_h += X[:, None] * hidden_error_term
        # Weight step (hidden to output)
        delta_weights_h_o += hidden_outputs[:, None] * output_error_term

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # Update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        # Update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records
        # Delta contains weight updates for every record,
        # so I am dividing the update by n to make the update an average for all records

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Forward pass ####
        ## Hidden layer
        # Signals into hidden layer
        hidden_inputs = features.dot(self.weights_input_to_hidden)
        # Signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        ## Output layer
        # Signals into final output layer
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output)
        # Signals from final output layer
        final_outputs = final_inputs

        return final_outputs


#########################################################
# Set hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.4
hidden_nodes = 7
output_nodes = 1
