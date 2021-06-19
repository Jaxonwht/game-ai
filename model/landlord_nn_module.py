from typing import Tuple

import torch
import torch.nn as nn


class LandLordNN(nn.Module):
    conv1_kernel: Tuple[int, int] = (2, 2)
    conv2_kernel: Tuple[int, int] = (2, 2)
    conv3_kernel: Tuple[int, int] = (2, 2)

    def __init__(self, channel_number: int, row_size: int, col_size: int, output_size: int) -> None:
        # pylint: disable=too-many-locals
        super().__init__()
        conv1: nn.Conv2d = nn.Conv2d(1, channel_number, LandLordNN.conv1_kernel)
        row_size_temp: int = LandLordNN.output_dimension(row_size, LandLordNN.conv1_kernel[0])
        col_size_temp: int = LandLordNN.output_dimension(col_size, LandLordNN.conv1_kernel[1])

        leaky_relu1: nn.LeakyReLU = nn.LeakyReLU()

        conv2: nn.Conv2d = nn.Conv2d(channel_number, channel_number, LandLordNN.conv2_kernel)
        row_size_temp = LandLordNN.output_dimension(row_size_temp, LandLordNN.conv2_kernel[0])
        col_size_temp = LandLordNN.output_dimension(col_size_temp, LandLordNN.conv2_kernel[1])

        leaky_relu2: nn.LeakyReLU = nn.LeakyReLU()

        conv3: nn.Conv2d = nn.Conv2d(channel_number, 1, LandLordNN.conv3_kernel)
        row_size_temp = LandLordNN.output_dimension(row_size_temp, LandLordNN.conv3_kernel[0])
        col_size_temp = LandLordNN.output_dimension(col_size_temp, LandLordNN.conv3_kernel[1])

        leaky_relu3: nn.LeakyReLU = nn.LeakyReLU()

        flatten: nn.Flatten = nn.Flatten(1)

        input_temp: int = row_size_temp * col_size_temp

        linear: nn.Linear = nn.Linear(input_temp, output_size)

        leaky_relu4 = nn.LeakyReLU()

        self.model = nn.Sequential(
            conv1,
            leaky_relu1,
            conv2,
            leaky_relu2,
            conv3,
            leaky_relu3,
            flatten,
            linear,
            leaky_relu4
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)

    @staticmethod
    def output_dimension(input_dim: int, kernel_size: int) -> int:
        return input_dim - kernel_size + 1
