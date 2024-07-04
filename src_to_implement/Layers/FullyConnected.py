import numpy as np
from Layers import Base

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):

        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        # self.biases = np.random.uniform(0, 1, (1, output_size))

        self.optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        biases = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)
        self.weights = np.vstack((weights, biases))

    def forward(self, input_tensor):
        biases_tensor_ones = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.hstack((input_tensor, biases_tensor_ones))
        output_tensor = np.dot( self.input_tensor, self.weights)
        return output_tensor

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # self.gradient_biases = np.sum(error_tensor, axis=0, keepdims=True)
        error_tensor_prev = np.dot(error_tensor, self.weights[:-1].T)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return error_tensor_prev

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    # def get_optimizer(self):
    #     return self.optimizer
    #
    # def set_optimizer(self, optimizer):
    #     self.optimizer = optimizer
    #
    # def get_gradient_weights(self):
    #     return self.gradient_weights
    #
    # def set_gradient_weights(self, gradient_weights):
    #     self.gradient_weights = gradient_weights
