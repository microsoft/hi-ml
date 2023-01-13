#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import torch

from health_multimodal.image.model.resnet import resnet18, resnet50


@torch.no_grad()
def test_return_intermediate_layers() -> None:
    """Check if the ``return_intermediate_layers`` keyword argument works correctly."""

    image = torch.rand(1, 3, 480, 480)
    x0_dims = batch_size, num_channels, height, width = 1, 64, 120, 120

    # Check resnet18 intermediate layers
    model = resnet18(pretrained=False, progress=False)
    outputs = model(image, return_intermediate_layers=True)
    assert outputs[0].shape == x0_dims

    layer_dims = batch_size, num_channels, height, width
    for i in range(1, len(outputs)):
        assert outputs[i].shape == layer_dims
        layer_dims = batch_size, layer_dims[1] * 2, layer_dims[2] // 2, layer_dims[3] // 2

    # Check resnet50 intermediate layers
    model = resnet50(pretrained=False, progress=False)
    outputs = model(image, return_intermediate_layers=True)
    assert outputs[0].shape == x0_dims

    layer_dims = batch_size, 4 * num_channels, height, width
    for i in range(1, len(outputs)):
        assert outputs[i].shape == layer_dims
        layer_dims = batch_size, layer_dims[1] * 2, layer_dims[2] // 2, layer_dims[3] // 2
