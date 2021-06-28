from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LandLordNNv2(nn.Module):
    conv1_kernel: Tuple[int, int] = (2, 2)
    conv2_kernel: Tuple[int, int] = (2, 2)
    conv3_kernel: Tuple[int, int] = (2, 2)

    def __init__(
        self,
        channel_number: int,
        row_size: int,
        col_size: int,
        history_output_row_size: int,
        history_output_col_size: int,
        output_size: int
    ) -> None:
        # pylint: disable=too-many-locals, too-many-arguments
        super().__init__()
        self.direct_row_size = row_size
        conv1: nn.Conv2d = nn.Conv2d(1, channel_number, LandLordNNv2.conv1_kernel)
        row_size_temp: int = LandLordNNv2.output_dimension(row_size, LandLordNNv2.conv1_kernel[0])
        col_size_temp: int = LandLordNNv2.output_dimension(col_size, LandLordNNv2.conv1_kernel[1])

        leaky_relu1: nn.LeakyReLU = nn.LeakyReLU()

        conv2: nn.Conv2d = nn.Conv2d(channel_number, channel_number, LandLordNNv2.conv2_kernel)
        row_size_temp = LandLordNNv2.output_dimension(row_size_temp, LandLordNNv2.conv2_kernel[0])
        col_size_temp = LandLordNNv2.output_dimension(col_size_temp, LandLordNNv2.conv2_kernel[1])

        leaky_relu2: nn.LeakyReLU = nn.LeakyReLU()

        conv3: nn.Conv2d = nn.Conv2d(channel_number, 1, LandLordNNv2.conv3_kernel)
        row_size_temp = LandLordNNv2.output_dimension(row_size_temp, LandLordNNv2.conv3_kernel[0])
        col_size_temp = LandLordNNv2.output_dimension(col_size_temp, LandLordNNv2.conv3_kernel[1])

        leaky_relu3: nn.LeakyReLU = nn.LeakyReLU()

        flatten: nn.Flatten = nn.Flatten(1)

        input_temp_direct: int = row_size_temp * col_size_temp

        self.model_direct = nn.Sequential(
            conv1,
            leaky_relu1,
            conv2,
            leaky_relu2,
            conv3,
            leaky_relu3,
            flatten,
        )

        adaptive_pool = nn.AdaptiveAvgPool2d((history_output_row_size, history_output_col_size))

        conv1 = nn.Conv2d(1, channel_number, LandLordNNv2.conv1_kernel)
        row_size_temp = LandLordNNv2.output_dimension(history_output_row_size, LandLordNNv2.conv1_kernel[0])
        col_size_temp = LandLordNNv2.output_dimension(history_output_col_size, LandLordNNv2.conv1_kernel[1])

        leaky_relu1 = nn.LeakyReLU()

        conv2 = nn.Conv2d(channel_number, channel_number, LandLordNNv2.conv2_kernel)
        row_size_temp = LandLordNNv2.output_dimension(row_size_temp, LandLordNNv2.conv2_kernel[0])
        col_size_temp = LandLordNNv2.output_dimension(col_size_temp, LandLordNNv2.conv2_kernel[1])

        leaky_relu2 = nn.LeakyReLU()

        conv3 = nn.Conv2d(channel_number, 1, LandLordNNv2.conv3_kernel)
        row_size_temp = LandLordNNv2.output_dimension(row_size_temp, LandLordNNv2.conv3_kernel[0])
        col_size_temp = LandLordNNv2.output_dimension(col_size_temp, LandLordNNv2.conv3_kernel[1])

        leaky_relu3 = nn.LeakyReLU()

        flatten = nn.Flatten(1)

        self.model_history = nn.Sequential(
            adaptive_pool,
            conv1,
            leaky_relu1,
            conv2,
            leaky_relu2,
            conv3,
            leaky_relu3,
            flatten
        )

        input_temp_history = row_size_temp * col_size_temp

        self.linear_combine = nn.Linear(input_temp_direct + input_temp_history, output_size)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        direct = self.model_direct(input_tensor[:, :, :self.direct_row_size, :])
        history = self.model_history(input_tensor[:, :, self.direct_row_size:, :])
        raw_tensor = self.leaky_relu(self.linear_combine(torch.hstack((direct, history))))
        return torch.hstack((F.softmax(raw_tensor[:, :-1], dim=0), raw_tensor[:, -1].unsqueeze(1)))

    @staticmethod
    def output_dimension(input_dim: int, kernel_size: int) -> int:
        return input_dim - kernel_size + 1
