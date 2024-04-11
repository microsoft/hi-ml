#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Set, Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp, trunc_normal_


@dataclass
class MultiHeadAttentionOutput:
    mha_output: torch.Tensor
    attention: Optional[torch.Tensor] = None


class VisionTransformerPooler(nn.Module):
    """
    :param input_dim: Input feature dimension (i.e., channels in old CNN terminology)
    :param grid_shape: Shape of the grid of patches per image
    :param num_heads: Number of self-attention heads within the MHA block
    :param num_blocks: Number of blocks per attention layer
    :param norm_layer: Normalisation layer

    `self.type_embed`: Is used to characterise prior and current scans, and
                       create permutation variance across modalities/series.
    """

    def __init__(
        self,
        input_dim: int,
        grid_shape: Tuple[int, int],
        num_heads: int = 8,
        num_blocks: int = 3,
        norm_layer: Any = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        block_kwargs = dict(
            dim=input_dim,
            num_heads=num_heads,
            mlp_ratio=1.0,
            drop=0.10,
            attn_drop=0.10,
            drop_path=0.25,
            act_layer=nn.GELU,
            norm_layer=norm_layer,
        )
        self.blocks = nn.ModuleList([Block(**block_kwargs) for _ in range(num_blocks)])
        self.norm_post = norm_layer(input_dim)
        self.grid_shape = grid_shape
        self.num_patches = grid_shape[0] * grid_shape[1]
        self.num_blocks = num_blocks

        # Temporal positional embeddings
        num_series: int = 2
        self.type_embed = nn.Parameter(torch.zeros(num_series, 1, input_dim))
        trunc_normal_(self.type_embed, std=0.02)

        # Positional embeddings 1 x L x C (L: Sequence length, C: Feature dimension)
        self.pos_drop = nn.Dropout(p=0.10)
        pos_embed_class = SinePositionEmbedding(embedding_dim=input_dim // 2, normalize=True)
        pos_embed = pos_embed_class(mask=torch.ones([1, grid_shape[0], grid_shape[1]]))  # 1 x L x C
        self.register_buffer("pos_embed", pos_embed, persistent=False)

        # Initialisation
        self.apply(self._init_weights)

    def no_weight_decay(self) -> Set[str]:
        return {'type_embed'}

    def forward(self, current_image: torch.Tensor, previous_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = current_image.shape
        assert H == self.grid_shape[0] and W == self.grid_shape[1], "Input and grid shapes do not match"

        # Flatten patch embeddings to have shape (B x L x C), L = H * W
        if previous_image is not None:
            assert previous_image.shape == current_image.shape, "current_image and previous_image shapes do not match"
            previous_image = previous_image.view(B, C, H * W).transpose(1, 2)
        current_image = current_image.view(B, C, H * W).transpose(1, 2)
        pos_embed = self.pos_embed.repeat(B, 1, 1)  # type: ignore

        # Final token activations (B x 2L x C)
        token_features = self.forward_after_reshape(x=current_image, pos_embed=pos_embed, x_previous=previous_image)

        # Extract the patch features of current image
        cur_img_token_id = 0
        current_token_features = token_features[:, cur_img_token_id : self.num_patches + cur_img_token_id]
        current_patch_features = current_token_features.transpose(1, 2).view(B, C, H, W)

        return current_patch_features

    def forward_after_reshape(
        self, x: torch.Tensor, pos_embed: torch.Tensor, x_previous: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, _ = x.shape  # Batch, Sequence length, Feature dimension

        # Positional and type embeddings
        type_embed = self.type_embed[0].expand(B, L, -1)
        if x_previous is not None:
            x = torch.cat((x, x_previous), dim=1)
            pos_embed = torch.cat((pos_embed, pos_embed), dim=1)
            prev_type_embed = self.type_embed[1].expand(B, L, -1)
            type_embed = torch.cat((type_embed, prev_type_embed), dim=1)

        # Add positional and type embeddings (used in query and key matching)
        pos_and_type_embed = pos_embed + type_embed

        # Positional dropout
        x = self.pos_drop(x)

        # Multihead attention followed by MLP
        for block in self.blocks:
            x = block(x=x, pos_and_type_embed=pos_and_type_embed)
        x = self.norm_post(x)

        return x

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head self attention module

    The content builds on top of the TIMM library (vision_transformer.py) and differs by the following:
        - Defines a custom `MultiHeadAttentionLayer` which does not only apply `self-attention` but it can be
            generalised to arbitrary (query, key, value) input tuples. This feature can be valuable to process
            more than 2 scans at a time.
        - `Self-attention` specific use-case can still be invoked by calling the `forward_as_mhsa` method.
    """

    def __init__(
        self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"The embedding dim ({dim}) must be divisible by the number of heads ({num_heads})"
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.return_attention = False

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> MultiHeadAttentionOutput:
        B, N, C = v.shape
        assert (
            C % self.num_heads == 0
        ), f"The embedding dim ({C}) must be divisible by the number of heads ({self.num_heads})"

        w_q = self.proj_q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        w_k = self.proj_k(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        w_v = self.proj_v(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (w_q @ w_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        o = (attn @ w_v).transpose(1, 2).reshape(B, N, C)
        o = self.proj(o)
        o = self.proj_drop(o)

        attention_output = attn if self.return_attention else None

        return MultiHeadAttentionOutput(mha_output=o, attention=attention_output)

    def forward_as_mhsa(self, input: torch.Tensor) -> MultiHeadAttentionOutput:
        return self(k=input, q=input, v=input)


class Block(nn.Module):
    """
    Encapsulates multi-layer perceptron and multi-head self attention modules into a block.

    The content builds on top of the TIMM library (vision_transformer.py) and differs by the following:
        - This implementation uses spatio-temporal positional embeddings instead of 2D positional embeddings only,
            and they are taken into account within the forward pass of each ViT block.
        - Utilises the custom defined `MultiHeadAttentionLayer` which does not apply `self-attention` only but can be
            generalised to arbitrary (query, key, value) tuples. This can be valuable to process more than 2 scans.

    Positional and type embeddings are handled in a similar fashion as DETR object localisation paper
    https://alcinos.github.io/detr_page/, where a fixed set of sine/cos positional embeddings are used
    in an additive manner to Q and K tensors.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 1.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttentionLayer(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def with_pos_and_type_embed(self, tensor: torch.Tensor, emb: Optional[torch.Tensor]) -> torch.Tensor:
        # Add positional embeddings to key and query tensors
        return tensor if emb is None else tensor + emb

    def forward(self, x: torch.Tensor, pos_and_type_embed: Optional[torch.Tensor]) -> torch.Tensor:
        x_with_emb = self.with_pos_and_type_embed(self.norm1(x), emb=pos_and_type_embed)
        x = x + self.drop_path(self.attn.forward_as_mhsa(x_with_emb).mha_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SinePositionEmbedding:
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, embedding_dim: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        assert mask is not None, "No pixel mask provided"
        B, H, W = mask.shape
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).view(B, H * W, self.embedding_dim * 2)

        return pos
