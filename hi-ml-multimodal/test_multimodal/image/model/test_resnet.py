#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import torch

from health_multimodal.image.model.resnet import resnet18, resnet50


@torch.no_grad()
def test_return_intermediate_layers() -> None:
    """
    Checks if the ``return_intermediate_layers`` argument works correctly
    """

    image = torch.rand(1, 3, 480, 480)
    N, C, H, W = 1, 64, 120, 120
    x0_dims = (N, C, H, W)

    # Check resnet18 intermediate layers
    model = resnet18(pretrained=False, progress=False)
    outputs = model(image, return_intermediate_layers=True)
    assert outputs[0].shape == x0_dims

    layer_dims = (N, C, H, W)
    for i in range(1, len(outputs)):
        assert outputs[i].shape == layer_dims
        layer_dims = (N, layer_dims[1] * 2, layer_dims[2] // 2, layer_dims[3] // 2)

    # Check resnet50 intermediate layers
    model = resnet50(pretrained=False, progress=False)
    outputs = model(image, return_intermediate_layers=True)
    assert outputs[0].shape == x0_dims

    expansion = 4
    layer_dims = (N, C * expansion, H, W)
    for i in range(1, len(outputs)):
        assert outputs[i].shape == layer_dims
        layer_dims = (N, layer_dims[1] * 2, layer_dims[2] // 2, layer_dims[3] // 2)
