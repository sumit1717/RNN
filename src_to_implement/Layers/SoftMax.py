from Layers import Base
import numpy as np


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        exp_input_tensor = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.softmax_output_tensor = exp_input_tensor / np.sum(exp_input_tensor, axis=1, keepdims=True)
        return self.softmax_output_tensor

    def backward(self, error_tensor):
        return self.softmax_output_tensor * (error_tensor - np.sum(error_tensor * self.softmax_output_tensor, axis=1, keepdims=True))
