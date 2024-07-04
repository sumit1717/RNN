import numpy as np


class Constant:
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant_value)


class UniformRandom:
    def __init__(self, lower_bound=0, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(self.lower_bound, self.upper_bound, size=weights_shape)


class Xavier:
    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in, fan_out):
        std_dev = np.sqrt(2/(fan_in + fan_out))
        return np.random.normal(0, std_dev, size=weights_shape)


class He:
    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in, fan_out):
        std_dev = np.sqrt(2/fan_in)
        return np.random.normal(0, std_dev, size=weights_shape)