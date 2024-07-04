import copy

import numpy as np
from Layers import Base
from scipy.signal import correlate, convolve, convolve2d, correlate2d


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape
        self.num_kernels = num_kernels
        self.convolution_shape = convolution_shape

        if len(convolution_shape) == 2:
            self.is_1d = True
        elif len(convolution_shape) == 3:
            self.is_1d = False

        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self.trainable = True
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      self.num_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        channels = input_tensor.shape[1]
        self.input_tensor = input_tensor

        output_tensor = np.zeros([batch_size, self.num_kernels, *input_tensor.shape[2:]])

        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(channels):
                    if self.is_1d:  # 1D case
                        output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode="same")
                    else:  # 2D case
                        output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode="same")
                output_tensor[b, k] += self.bias[k]

        # Apply stride
        if self.is_1d:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0]]
        else:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        self.output_tensor = output_tensor
        return output_tensor

    def upsampling(self, input_tensor, output_shape):
        result = np.zeros(output_shape)
        if self.is_1d:
            result[::self.stride_shape[0]] = input_tensor
            return result
        else:
            result[::self.stride_shape[0], ::self.stride_shape[1]] = input_tensor
            return result

    def backward(self, error_tensor):
        self.gradient_weights = np.zeros_like(self.weights)
        if self.is_1d:
            self.gradient_bias = np.sum(error_tensor, (0, 2))
        else:
            self.gradient_bias = np.sum(error_tensor, (0, 2, 3))

        gradient_input_tensor = np.zeros_like(self.input_tensor)

        filter_dims = np.array(self.convolution_shape[1:])
        pad_start = filter_dims // 2
        pad_end = filter_dims - pad_start - 1
        if self.is_1d:
            pad_width = [(0, 0), (0, 0), (pad_start[0], pad_end[0])]
        else:
            pad_width = [(0, 0), (0, 0), (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1])]

        padded_input_tensor = np.pad(self.input_tensor, pad_width=pad_width, constant_values=0)

        for batch in range(self.input_tensor.shape[0]):
            for kernel in range(self.num_kernels):
                for channel in range(self.input_tensor.shape[1]):
                    up_sampled_error = self.upsampling(error_tensor[batch, kernel], self.input_tensor.shape[2:])
                    grad_weights = correlate(padded_input_tensor[batch, channel], up_sampled_error, "valid")
                    self.gradient_weights[kernel, channel] += grad_weights

                    grad_input = convolve(up_sampled_error, self.weights[kernel, channel], "same")
                    gradient_input_tensor[batch, channel] += grad_input

        if self._optimizer is not None:
            self.weights = copy.deepcopy(self._optimizer).calculate_update(self.weights, self.gradient_weights)
            self.bias = copy.deepcopy(self._optimizer).calculate_update(self.bias, self.gradient_bias)

        return gradient_input_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

