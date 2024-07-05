import numpy as np
from Layers import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.activation_output = 1 / (1 + np.exp(-input_tensor))
        return self.activation_output

    def backward(self, error_tensor):
        return error_tensor * self.activation_output * (1 - self.activation_output)