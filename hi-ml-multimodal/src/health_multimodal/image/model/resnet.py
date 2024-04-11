#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Any, List, Tuple, Type, Union

import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.resnet import ResNet50_Weights

TypeSkipConnections = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ResNetHIML(ResNet):
    """Wrapper class of the original torchvision ResNet model.

    The forward function is updated to return the penultimate layer
    activations, which are required to obtain image patch embeddings.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def forward(
        self, x: torch.Tensor, return_intermediate_layers: bool = False
    ) -> Union[torch.Tensor, TypeSkipConnections]:
        """ResNetHIML forward pass. Optionally returns intermediate layers using the
        ``return_intermediate_layers`` argument.

        :param return_intermediate_layers: If ``True``, return layers x0-x4 as a tuple,
            otherwise return x4 only.
        """

        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if return_intermediate_layers:
            return x0, x1, x2, x3, x4
        else:
            return x4


def _resnet(
    url: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNetHIML:
    """Instantiate a custom :class:`ResNet` model.

    Adapted from :mod:`torchvision.models.resnet`.
    """
    model = ResNetHIML(block=block, layers=layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(url, progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetHIML:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    :param pretrained: If ``True``, returns a model pre-trained on ImageNet.
    :param progress: If ``True``, displays a progress bar of the download to ``stderr``.
    """
    url = ResNet18_Weights.IMAGENET1K_V1.url
    return _resnet(url, BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetHIML:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    :param pretrained: If ``True``, returns a model pre-trained on ImageNet
    :param progress: If ``True``, displays a progress bar of the download to ``stderr``.
    """
    url = ResNet50_Weights.IMAGENET1K_V1.url
    return _resnet(url, Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
