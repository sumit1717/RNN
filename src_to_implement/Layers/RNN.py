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
        self.states = []

        self._optimizer = None

        self.tanh_activation = TanH()
        self.sigmoid_activation = Sigmoid()

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
        self.fc_input_hidden.weights = weights

    @property
    def gradient_weights(self):
        return self.fc_input_hidden.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.fc_input_hidden.gradient_weights = gradient_weights

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_input_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_hidden_output.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        batch_size = input_tensor.shape[0]
        self.states = []

        if not self.memorize:
            self.previous_hidden_state = np.zeros(self.hidden_size)

        output_tensor = np.zeros([batch_size, self.output_size])
        for time_step in range(batch_size):
            local_states = []
            current_input = input_tensor[time_step]

            combined_input = np.concatenate((current_input, self.previous_hidden_state))
            combined_input = np.expand_dims(combined_input, axis=0)

            hidden_unactivated = self.fc_input_hidden.forward(combined_input)
            local_states.append(self.fc_input_hidden.input_tensor)

            hidden_activated = self.tanh_activation.forward(hidden_unactivated)
            local_states.append(self.tanh_activation.activation_output)

            output_y_unactivated = self.fc_hidden_output.forward(hidden_activated)
            local_states.append(self.fc_hidden_output.input_tensor)

            output_y = self.sigmoid_activation.forward(output_y_unactivated)
            local_states.append(self.sigmoid_activation.activation_output)

            self.previous_hidden_state = hidden_activated.flatten()
            self.states.append(local_states)

            output_tensor[time_step] = output_y

        return output_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        gradient_input_tensor = np.zeros_like(self.input_tensor)
        previous_hidden_gradient = np.zeros(self.hidden_size)

        accumulated_gradient_weights_input_hidden = np.zeros_like(self.fc_input_hidden.weights)
        accumulated_gradient_weights_hidden_output = np.zeros_like(self.fc_hidden_output.weights)

        for time_step in reversed(range(batch_size)):

            current_error = error_tensor[time_step]
            combined_input, hidden_activated, output_y_unactivated, output_y = self.states[time_step]

            self.sigmoid_activation.activation_output = output_y
            delta_output = self.sigmoid_activation.backward(current_error)

            self.fc_hidden_output.input_tensor = output_y_unactivated

            error_hidden_state = self.fc_hidden_output.backward(delta_output) + previous_hidden_gradient

            self.tanh_activation.activation_output = hidden_activated
            delta_hidden_state = self.tanh_activation.backward(error_hidden_state)

            self.fc_input_hidden.input_tensor = combined_input

            previous_hidden_gradient = self.fc_input_hidden.backward(delta_hidden_state)[:, self.input_size:]

            if self.fc_hidden_output.gradient_weights is not None:
                accumulated_gradient_weights_hidden_output += self.fc_hidden_output.gradient_weights.astype(np.float64)

            if self.fc_input_hidden.gradient_weights is not None:
                accumulated_gradient_weights_input_hidden += self.fc_input_hidden.gradient_weights.astype(np.float64)

            gradient_input_tensor[time_step] = self.fc_input_hidden.backward(delta_hidden_state)[0, :self.input_size]

        self.gradient_weights = accumulated_gradient_weights_input_hidden

        if self._optimizer:
            self.fc_input_hidden.weights = self.optimizer_fc_input_hidden.calculate_update(
                self.fc_input_hidden.weights, accumulated_gradient_weights_input_hidden)
            self.fc_hidden_output.weights = self.optimizer_fc_hidden_output.calculate_update(
                self.fc_hidden_output.weights, accumulated_gradient_weights_hidden_output)

        return gradient_input_tensor

    def calculate_regularization_loss(self):
        loss = 0
        if self.fc_input_hidden.optimizer:
            loss += self.fc_input_hidden.optimizer.calculate_regularization_loss(self.fc_input_hidden.weights)
        if self.fc_hidden_output.optimizer:
            loss += self.fc_hidden_output.optimizer.calculate_regularization_loss(self.fc_hidden_output.weights)
        return loss