#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Callable, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Fully connected layers to map between image embeddings and projection space where pairs of images are compared.

    :param input_dim: Input embedding feature size
    :param hidden_dim: Hidden layer size in MLP
    :param output_dim: Output projection size
    :param use_1x1_convs: Use 1x1 conv kernels instead of 2D linear transformations for speed and memory efficiency.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, use_1x1_convs: bool = False
    ) -> None:
        super().__init__()

        if use_1x1_convs:
            linear_proj_1_args = {'in_channels': input_dim, 'out_channels': hidden_dim, 'kernel_size': 1, 'bias': False}
            linear_proj_2_args = {'in_channels': hidden_dim, 'out_channels': output_dim, 'kernel_size': 1, 'bias': True}
            normalisation_layer: Callable = nn.BatchNorm2d
            projection_layer: Callable = nn.Conv2d
        else:
            linear_proj_1_args = {'in_features': input_dim, 'out_features': hidden_dim, 'bias': False}
            linear_proj_2_args = {'in_features': hidden_dim, 'out_features': output_dim, 'bias': True}
            normalisation_layer = nn.BatchNorm1d
            projection_layer = nn.Linear

        self.output_dim = output_dim
        self.input_dim = input_dim
        if hidden_dim is not None:
            self.model = nn.Sequential(
                projection_layer(**linear_proj_1_args),
                normalisation_layer(hidden_dim),
                nn.ReLU(inplace=True),
                projection_layer(**linear_proj_2_args),
            )
        else:
            self.model = nn.Linear(input_dim, output_dim)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the multi-layer perceptron"""
        x = self.model(x)
        return x


class MultiTaskModel(nn.Module):
    """Torch module for multi-task classification heads. We create a separate classification head
    for each task and perform a forward pass on each head independently in forward(). Classification
    heads are instances of `MLP`.

    :param input_dim: Number of dimensions of the input feature map.
    :param classifier_hidden_dim: Number of dimensions of hidden features in the MLP.
    :param num_classes: Number of output classes per task.
    :param num_tasks: Number of classification tasks or heads required.
    """

    def __init__(self, input_dim: int, classifier_hidden_dim: Optional[int], num_classes: int, num_tasks: int):
        super().__init__()

        self.num_classes = num_classes
        self.num_tasks = num_tasks

        for task in range(num_tasks):
            # TODO check if softmax not needed here.
            setattr(self, "fc_" + str(task), MLP(input_dim, output_dim=num_classes, hidden_dim=classifier_hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [batch_size, num_tasks, num_classes] tensor of logits."""
        batch_size = x.shape[0]
        out = torch.zeros((batch_size, self.num_classes, self.num_tasks), dtype=x.dtype, device=x.device)
        for task in range(self.num_tasks):
            classifier = getattr(self, "fc_" + str(task))
            out[:, :, task] = classifier(x)
        return out
