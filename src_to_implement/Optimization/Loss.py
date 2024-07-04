import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = np.finfo(np.float64).eps

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor

        return np.sum(-1 * label_tensor * np.log(prediction_tensor + self.epsilon))

    def backward(self, label_tensor):
        return -label_tensor / (self.prediction_tensor + self.epsilon)
