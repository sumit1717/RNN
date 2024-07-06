import copy
import numpy as np
from Layers import Base
from Layers import Helpers


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.testing_phase = False

        self.channels = channels
        self.epsilon = np.finfo(float).eps
        self.momentum = 0.8

        self.gamma = np.ones(channels)
        self.beta = np.zeros(channels)

        self.moving_mean = None
        self.moving_variance = None

        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))

    def forward(self, input_tensor):
        is_convolutional = (len(input_tensor.shape) == 4)

        if is_convolutional:
            input_tensor = self.reformat(input_tensor)

        self.input_tensor = input_tensor

        if self.testing_phase:
            self.batch_mean = self.moving_mean
            self.batch_variance = self.moving_variance
        else:
            batch_mean = np.mean(input_tensor, axis=0)
            batch_variance = np.var(input_tensor, axis=0)

            self.batch_mean = batch_mean
            self.batch_variance = batch_variance

            if self.moving_mean is None:
                self.moving_mean = batch_mean
                self.moving_variance = batch_variance
            else:
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean
                self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * batch_variance

        self.normalized_input = (input_tensor - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)
        output_tensor = self.gamma * self.normalized_input + self.beta

        if is_convolutional:
            output_tensor = self.reformat(output_tensor)

        return output_tensor

    def backward(self, error_tensor):

        is_convolutional = (len(error_tensor.shape) == 4)
        if is_convolutional:
            error_tensor = self.reformat(error_tensor)

        gradient_beta = np.sum(error_tensor, axis=0)
        gradient_gamma = np.sum(error_tensor * self.normalized_input, axis=0)
        gradient_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.gamma,
                                              self.batch_mean, self.batch_variance)

        if self._optimizer is not None:
            self.gamma = self._optimizer.weight.calculate_update(self.gamma, gradient_gamma)
            self.beta = self._optimizer.bias.calculate_update(self.beta, gradient_gamma)

        if is_convolutional:
            gradient_input = self.reformat(gradient_input)

        self.gradient_weights = gradient_gamma
        self.gradient_bias = gradient_beta

        return gradient_input

    def reformat(self, tensor):
        if tensor.ndim == 4:
            batch_size, channels, height, width = tensor.shape
            self.reformat_shape = tensor.shape
            tensor = tensor.reshape(batch_size, channels, height * width)
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(batch_size * height * width, channels)
        else:
            batch_size, channels, height, width = self.reformat_shape
            tensor = tensor.reshape(batch_size, height * width, channels)
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(batch_size, channels, height, width)
        return tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = optimizer
        self._optimizer.bias = optimizer

    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, weights_value):
        self.gamma = weights_value

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, bias_value):
        self.beta = bias_value