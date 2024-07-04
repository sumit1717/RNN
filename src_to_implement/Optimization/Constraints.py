import numpy as np


class L2_Regularizer:
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def calculate_gradient(self, weights):
        grad = self.alpha * weights
        return grad

    def norm(self, weights):
        norm = self.alpha * np.sum(weights**2)
        return norm


class L1_Regularizer:
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def calculate_gradient(self, weights):
        grad = self.alpha * np.sign(weights)
        return grad

    def norm(self, weights):
        norm = self.alpha * np.sum(np.abs(weights))
        return norm
