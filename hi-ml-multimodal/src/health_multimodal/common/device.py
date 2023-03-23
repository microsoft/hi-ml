#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


import torch


def get_module_device(module: torch.nn.Module) -> torch.device:
    """
    Returns the device of the module
    """
    device = next(module.parameters()).device  # type: ignore
    assert isinstance(device, torch.device)

    return device
