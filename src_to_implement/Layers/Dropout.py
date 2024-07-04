import numpy as np
from Layers import Base

class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.dropout_mask_rand = []
        self.trainable = False
        self.testing_phase = False

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor

        dropout_mask_rand = (np.random.rand(*input_tensor.shape) < self.probability) / self.probability
        self.dropout_mask_rand = dropout_mask_rand

        return input_tensor * dropout_mask_rand

    def backward(self, error_tensor):
        return error_tensor * self.dropout_mask_rand
