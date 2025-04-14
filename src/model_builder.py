from enum import Enum
from dataclasses import dataclass
from typing import Any


import torch
import torch.nn as nn


class ModelType(Enum):
    InceptionBlock = "inception"


@dataclass
class ModelHyperParameters:
    input_size: int
    intermediate_size: dict[str, int]
    output_size: dict[str, int]
    activation_func: Any


def create_model(
    model_type: ModelType, model_hyperparameters: ModelHyperParameters
) -> nn.Module:
    match model_type:
        case ModelType.InceptionBlock:
            return InceptionBlock(model_hyperparameters)


class InceptionBlock(nn.Module):
    def __init__(self, hyperparameters: ModelHyperParameters) -> None:
        super().__init__()
        input_size = hyperparameters.input_size
        intermediate_size = hyperparameters.intermediate_size
        output_size = hyperparameters.output_size
        activation_func = hyperparameters.activation_func

        self.conv1x1: nn.Module = nn.Sequential(
            nn.Conv2d(input_size, output_size["1x1"], kernel_size=1),
            nn.BatchNorm2d(output_size["1x1"]),
            activation_func,
        )

        self.conv3x3: nn.Module = nn.Sequential(
            nn.Conv2d(input_size, intermediate_size["3x3"], kernel_size=1),
            nn.Conv2d(intermediate_size["3x3"], output_size["3x3"], kernel_size=3),
            nn.BatchNorm2d(output_size["3x3"]),
            activation_func,
        )

        self.conv5x5: nn.Module = nn.Sequential(
            nn.Conv2d(input_size, intermediate_size["5x5"], kernel_size=1),
            nn.Conv2d(intermediate_size["5x5"], output_size["5x5"], kernel_size=3),
            nn.BatchNorm2d(output_size["5x5"]),
            activation_func,
        )

        self.max_pool: nn.Module = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(input_size, output_size["max_pool"], kernel_size=1),
            nn.BatchNorm2d(output_size["max_pool"]),
            activation_func,
        )

    def forward(self, input):
        output1x1 = self.conv1x1(input)
        output3x3 = self.conv3x3(input)
        output5x5 = self.conv5x5(input)
        output_max_pool = self.max_pool(input)
        output_total = torch.cat(
            (output1x1, output3x3, output5x5, output_max_pool), dim=1
        )
        return output_total
