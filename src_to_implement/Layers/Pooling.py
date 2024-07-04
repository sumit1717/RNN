import numpy as np
from Layers import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, num_channels, input_height, input_width = input_tensor.shape

        output_height = (input_height - self.pooling_shape[0]) // self.stride_shape[0] + 1
        output_width = (input_width - self.pooling_shape[1]) // self.stride_shape[1] + 1
        output_tensor = np.zeros((batch_size, num_channels, output_height, output_width))

        self.argmax_indices = np.zeros((batch_size, num_channels, output_height, output_width), dtype=int)

        for row in range(output_height):
            row_start = row * self.stride_shape[0]
            row_end = row_start + self.pooling_shape[0]
            for col in range(output_width):
                col_start = col * self.stride_shape[1]
                col_end = col_start + self.pooling_shape[1]

                pool_region = input_tensor[:, :, row_start:row_end, col_start:col_end]
                pool_region_flat = pool_region.reshape(batch_size, num_channels, -1)
                max_indices = np.argmax(pool_region_flat, axis=2)
                output_tensor[:, :, row, col] = pool_region_flat[
                    np.arange(batch_size)[:, None], np.arange(num_channels), max_indices]
                self.argmax_indices[:, :, row, col] = max_indices

        return output_tensor

    def backward(self, error_tensor):
        batch_size, num_channels, output_height, output_width = error_tensor.shape
        gradient_input = np.zeros_like(self.input_tensor)

        for row in range(output_height):
            row_start = row * self.stride_shape[0]
            for col in range(output_width):
                col_start = col * self.stride_shape[1]

                flat_indices = self.argmax_indices[:, :, row, col].flatten()
                error_flat = error_tensor[:, :, row, col].flatten()

                row_indices, col_indices = np.unravel_index(flat_indices, self.pooling_shape)
                row_indices += row_start
                col_indices += col_start

                batch_indices = np.repeat(np.arange(batch_size), num_channels)
                channel_indices = np.tile(np.arange(num_channels), batch_size)

                np.add.at(gradient_input, (batch_indices, channel_indices, row_indices, col_indices), error_flat)

        return gradient_input

