import numpy as np


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

