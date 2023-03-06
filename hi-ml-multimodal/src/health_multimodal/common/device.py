import torch


def get_module_device(module: torch.nn.Module) -> torch.device:
    """
    Returns the device of the module
    """
    device = next(module.parameters()).device  # type: ignore
    assert (isinstance(device, torch.device))

    return device
