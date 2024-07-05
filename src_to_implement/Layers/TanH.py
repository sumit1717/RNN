import numpy as np
from Layers import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.activation_output = np.tanh(input_tensor)
        return self.activation_output

    def backward(self, error_tensor):
        return error_tensor * (1 - np.square(self.activation_output))