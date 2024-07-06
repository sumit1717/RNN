import copy

import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self._memorize = False

        self.fc_input_hidden = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc_hidden_output = FullyConnected(self.hidden_size, self.output_size)

        self.previous_hidden_state = np.zeros(self.hidden_size)

        self._optimizer = None

        self.tanh_activation = TanH()
        self.sigmoid_activation = Sigmoid()

        self._gradient_weights = np.zeros(self.fc_input_hidden.weights.shape)
        self.weights = self.fc_input_hidden.weights

        self.gradient_weights_new = np.zeros(self.fc_input_hidden.weights.shape)


    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)
        self.optimizer_fc_input_hidden = copy.deepcopy(optimizer)
        self.optimizer_fc_hidden_output = copy.deepcopy(optimizer)

    @property
    def weights(self):
        return self.fc_input_hidden.weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
        self.fc_input_hidden.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_new

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.gradient_weights = gradient_weights
        self._gradient_weights = gradient_weights

        self.fc_hidden_output.gradient_weights = gradient_weights
        self.fc_input_hidden.gradient_weights = gradient_weights
        self.fc_hidden_output._gradient_weights = gradient_weights
        self.fc_input_hidden._gradient_weights = gradient_weights

    # @property
    # def gradient_weights(self):
    #     return self.gradient_weights_new
    #
    # @gradient_weights.setter
    # def gradient_weights(self, gradient_weights):
    #     self.gradient_weights_new = gradient_weights
    #     self.gradient_weights = self.fc_input_hidden.gradient_weights + self.fc_hidden_output.gradient_weights
    #     self._gradient_weights = gradient_weights
    #     self.fc_input_hidden.gradient_weights = gradient_weights
    #     self.fc_hidden_output.gradient_weights = gradient_weights

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_input_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_hidden_output.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        batch_size = input_tensor.shape[0]

        self.hidden_tensor = np.zeros((batch_size, self.hidden_size))
        self.outputs_tensor = np.zeros((batch_size, self.output_size))

        if not self.memorize:
            self.previous_hidden_state = np.zeros(self.hidden_size)

        for time_step in range(batch_size):
            current_input = input_tensor[time_step, ...]
            combined_input = np.concatenate((current_input, self.previous_hidden_state), axis=None).reshape(1, -1)
            hidden_unactivated = self.fc_input_hidden.forward(combined_input)
            hidden_activated = self.tanh_activation.forward(hidden_unactivated)
            self.hidden_tensor[time_step] = hidden_activated
            self.previous_hidden_state = hidden_activated
            output_y_unactivated = self.fc_hidden_output.forward(hidden_activated)
            output_y = self.sigmoid_activation.forward(output_y_unactivated)
            self.outputs_tensor[time_step] = output_y

        return self.outputs_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        gradient_input_tensor = np.zeros_like(self.input_tensor)
        previous_hidden_gradient = np.zeros(self.hidden_size)

        gradient_weights_input_hidden = np.zeros_like(self.fc_input_hidden.weights)
        gradient_weights_hidden_output = np.zeros_like(self.fc_hidden_output.weights)

        self.fc_input_hidden.gradient_weights = gradient_weights_input_hidden
        self.fc_hidden_output.gradient_weights = gradient_weights_hidden_output

        for time_step in reversed(range(batch_size)):
            current_error = error_tensor[time_step, ...]
            self.sigmoid_activation.activation_output = self.outputs_tensor[time_step]
            delta_output = self.sigmoid_activation.backward(current_error)
            gradient_weights_hidden_output += self.fc_hidden_output.gradient_weights

            error_hidden_state = self.fc_hidden_output.backward(delta_output.reshape(1,-1)) + previous_hidden_gradient
            self.tanh_activation.activation_output = self.hidden_tensor[time_step]
            delta_hidden_state = self.tanh_activation.backward(error_hidden_state)

            if time_step == 0:
                previous_hidden_state = np.zeros((1, self.hidden_size))
            else:
                previous_hidden_state = self.hidden_tensor[time_step]

            combined_input = np.concatenate((self.input_tensor[time_step], previous_hidden_state, [1]), axis=None).reshape(1, -1)
            self.fc_input_hidden.input_tensor = combined_input
            gradient_weights_input_hidden += self.fc_input_hidden.gradient_weights

            gradient_input_tensor[time_step] = self.fc_input_hidden.backward(delta_hidden_state)[:, :self.input_size]
            previous_hidden_gradient = self.fc_input_hidden.backward(delta_hidden_state)[:, self.input_size:]

        self.fc_input_hidden.gradient_weights = gradient_weights_input_hidden
        self.fc_hidden_output.gradient_weights = gradient_weights_input_hidden

        if self.optimizer:
            self.fc_input_hidden.weights = self.optimizer_fc_input_hidden.calculate_update(
                self.fc_input_hidden.weights, gradient_weights_input_hidden)
            self.fc_hidden_output.weights = self.optimizer_fc_hidden_output.calculate_update(
                self.fc_hidden_output.weights, gradient_weights_hidden_output)

        return gradient_input_tensor
