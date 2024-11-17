import numpy as np

class Layer:
    def __init__(self, neurons, activation_function, learning_rate, weight_num):
        self.neurons = int(neurons)
        self.activation_function = activation_function
        self.learning_rate = float(learning_rate)
        self.weights_num = weight_num
        self.differentiating = np.zeros(self.neurons)

        # Xavier Initialization for weights (recommended for tanh/sigmoid)
        self.W = np.random.randn(self.neurons, self.weights_num) * np.sqrt(2 / (self.weights_num + self.neurons))

        # Bias initialized to zeros
        self.bias = np.zeros(self.neurons)

        # Output of the neurons
        self.a_out = np.zeros(self.neurons)

        # Error for each neuron (used during backpropagation)
        self.error = np.zeros(self.neurons)

        # Additional parameters for activation functions (if necessary)
        self.a = 1
        self.b = 1
        
    # Activation function: Hyperbolic Tangent
    def hyperbolic_tangent(self, Z, b=1):
        # Using numpy to prevent overflow issues with large Z values
        return np.tanh(b * Z)
    
    # Derivative of Hyperbolic Tangent function
    def hyperbolic_tangent_differentiating(self, Z, a=1, b=1):
        # Using numpy to prevent overflow issues with large Z values
        return b * (1 - np.tanh(b * Z) ** 2)

    # Activation function selector
    def activation(self, Z, index, a=1):
        if self.activation_function.lower() == "sigmoid":
            self.differentiating[index] = self.sigmoid_differentiating(Z, a)
            return self.sigmoid(Z, a)
        elif self.activation_function.lower() == "hyperbolic_tangent":
            self.differentiating[index] = self.hyperbolic_tangent_differentiating(Z, a, b=1)
            return self.hyperbolic_tangent(Z, a)
        else:
            return self.softmax(Z)

    # Sigmoid activation and its derivative
    def sigmoid(self, Z, a=1):
        Z_clipped = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-self.a * Z_clipped))

    def sigmoid_differentiating(self, Z, a=1):
        Z_clipped = np.clip(Z, -500, 500)
        y = self.sigmoid(Z_clipped, self.a)
        return self.a * y * (1 - y)

    # Softmax function (for classification tasks)
    def softmax(self, Z):
        exps = np.exp(Z - np.max(Z))  # For numerical stability
        return exps / np.sum(exps)
