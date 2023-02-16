from typing import Callable, List, Optional
from timm.models.layers.helpers import to_2tuple
from timm.models import swin_tiny_patch4_window7_224
from torch import Tensor
import torch
import torch.nn as nn


class ConvStem(nn.Module):

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
    ) -> None:
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem: List[nn.Module] = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))

        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def get_ctranspath(tile_size: int = 224) -> nn.Module:
    model = swin_tiny_patch4_window7_224(pretrained=True, num_classes=0)
    model.patch_embed = ConvStem(img_size=tile_size, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm)
    return model


def get_pretrained_ctranspath(tile_size: int, checkpoint_path: str) -> nn.Module:
    model = get_ctranspath(tile_size)
    td = torch.load(checkpoint_path)
    model.load_state_dict(td['model'], strict=True)
    return model
