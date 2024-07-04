from Layers import Base


class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.cache_shape = None
        self.trainable = False

    def forward(self, input_tensor):
        self.cache_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.cache_shape)
