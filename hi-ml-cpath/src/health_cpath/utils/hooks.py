import torch.nn as nn

class ActivationHook(nn.Module):
    """Hook into encoder to get activations."""

    def __init__(self, model: nn.Module, layer_name: str = "layer4") -> None:
        """
        Initialize ActivationHook.

        Args:
            model (nn.Module): PyTorch model.
            layer_name (str, optional): Name of the layer to hook into. Defaults to "layer4".
        """
        super().__init__()
        self.acts_of_selected_layer = None

        self.model = model
        selected_layer = getattr(self.model.encoder.feature_extractor_fn, layer_name)
        self.activation_hook = selected_layer.register_forward_hook(self.forward_hook())

    def forward_hook(self):
        def hook(module, input, out):
            self.acts_of_selected_layer = out

        return hook

    def remove_hook(self) -> None:
        self.activation_hook.remove()
