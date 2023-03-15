#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
""" Layers for attention-based multi-instance learning (DeepMIL)
Created using the original DeepMIL paper and code from Ilse et al., 2018
https://github.com/AMLab-Amsterdam/AttentionDeepMIL (MIT License)
"""
from typing import Tuple, Optional
from torch import nn, Tensor, transpose, mm
import torch
import torch.nn.functional as F
from torch.nn import Module, TransformerEncoderLayer


class MeanPoolingLayer(nn.Module):
    """Mean pooling returns uniform weights and the average feature vector over the first axis"""

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        num_instances = features.shape[0]
        attention_weights = torch.full((1, num_instances), 1. / num_instances)
        pooled_features = features.mean(dim=0)
        pooled_features = pooled_features.view(1, -1)
        return (attention_weights, pooled_features)


class MaxPoolingLayer(nn.Module):
    """Max pooling returns uniform weights and the maximum feature vector over the first axis"""

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        num_instances = features.shape[0]
        pooled_features, indices = features.max(dim=0)
        frequency = torch.bincount(indices, minlength=num_instances)
        frequency_norm = frequency / sum(frequency)
        attention_weights = frequency_norm.view(1, num_instances)
        pooled_features = pooled_features.view(1, -1)
        return (attention_weights, pooled_features)


class AttentionLayer(nn.Module):
    """ AttentionLayer: Simple attention layer
    Requires size of input L, hidden D, and attention layers K (default K=1)
    """

    def __init__(self, input_dims: int,
                 hidden_dims: int,
                 attention_dims: int = 1) -> None:
        super().__init__()

        # Attention layers
        self.input_dims = input_dims                    # L
        self.hidden_dims = hidden_dims                  # D
        self.attention_dims = attention_dims            # K
        self.attention = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, self.attention_dims)
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        features = features.view(-1, self.input_dims)            # N x L
        attention_weights = self.attention(features)             # N x K
        attention_weights = transpose(attention_weights, 1, 0)   # K x N
        attention_weights = F.softmax(attention_weights, dim=1)  # Softmax over N : K x N
        pooled_features = mm(attention_weights, features)        # Matrix multiplication : K x L
        return attention_weights, pooled_features


class GatedAttentionLayer(nn.Module):
    """ GatedAttentionLayer: Gated attention layer
    Requires size of input L, hidden D, and output layers K (default K=1)
    """

    def __init__(self, input_dims: int,
                 hidden_dims: int,
                 attention_dims: int = 1) -> None:
        super().__init__()

        # Gated attention layers
        self.input_dims = input_dims                    # L
        self.hidden_dims = hidden_dims                  # D
        self.attention_dims = attention_dims            # K
        self.attention_V = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.hidden_dims, self.attention_dims)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        features = features.view(-1, self.input_dims)            # N x L
        A_V = self.attention_V(features)                         # N x D
        A_U = self.attention_U(features)                         # N x D
        attention_weights = self.attention_weights(A_V * A_U)    # Element-wise multiplication : N x K
        attention_weights = transpose(attention_weights, 1, 0)   # K x N
        attention_weights = F.softmax(attention_weights, dim=1)  # Softmax over N : K x N
        pooled_features = mm(attention_weights, features)        # Matrix multiplication : K x L
        return attention_weights, pooled_features


class CustomTransformerEncoderLayer(TransformerEncoderLayer):
    """Adaptation of the pytorch TransformerEncoderLayer that always outputs the attention weights.

    TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out, attention_weights = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out, attention_weights = encoder_layer(src)
    """
    # new forward returns output as well as attention weights

    def forward(self, src: torch.Tensor,  # type: ignore
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            sa_block_out, a = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + sa_block_out
            x = x + self._ff_block(self.norm2(x))
        else:
            sa_block_out, a = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + sa_block_out)
            x = self.norm2(x + self._ff_block(x))

        return x, a

    # new self-attention block, returns output as well as attention weights
    def _sa_block(self, x: Tensor,  # type: ignore
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, a = self.self_attn(x, x, x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=True)  # Just because of this flag I had to copy all of the code...
        x = x[0]
        return self.dropout1(x), a


class TransformerPooling(Module):
    """Create a Transformer encoder module consisting of multiple Transformer encoder layers.

    We use a additional classification token (cls token) for pooling like seen in ViT/Bert. First, the cls token is
    appended to the list of tiles encodings. Second, we perform self-attention between all tile encodings and the cls
    token. Last, we extract the cls token and use it for classification.

    Args:
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads per layer.
        dim_representation: Dimension of input encoding.
        transformer_dropout: The dropout value of transformer encoder layers.
    """

    def __init__(self, num_layers: int, num_heads: int, dim_representation: int, transformer_dropout: float) -> None:
        super(TransformerPooling, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_representation = dim_representation
        self.transformer_dropout = transformer_dropout
        self.cls_token = nn.Parameter(torch.zeros([1, dim_representation]))

        self.transformer_encoder_layers = []
        for _ in range(self.num_layers):
            self.transformer_encoder_layers.append(
                CustomTransformerEncoderLayer(self.dim_representation,
                                              self.num_heads,
                                              dim_feedforward=self.dim_representation,
                                              dropout=self.transformer_dropout,
                                              activation=F.gelu,
                                              batch_first=True))
        self.transformer_encoder_layers = torch.nn.ModuleList(self.transformer_encoder_layers)  # type: ignore

    def forward(self, features: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append cls token
        features = torch.vstack([self.cls_token, features]).unsqueeze(0)

        for i in range(self.num_layers):
            features, attention_weights = self.transformer_encoder_layers[i](features)

        # Extract cls token
        pooled_features = features[:, 0]

        # Get attention weights with respect to the cls token, without the element where it attends to itself

        self_attention_cls_token = attention_weights[0, 0, 0]  # type: ignore
        attention_weights = attention_weights[:, 0, 1:]  # type: ignore

        # We want A to sum to one, simple hack: add self_attention_cls_token/num_tiles to each element
        attention_weights += self_attention_cls_token / attention_weights.shape[-1]

        return (attention_weights, pooled_features)


class TransformerPoolingBenchmark(Module):
    """Create a Transformer encoder module consisting of multiple Transformer encoder layers.
    The pooling is inspired by the transformer pooling in `monai.networks.nets.milmodel`.
    The transformer pooling is used in the implementation of (Myronenko et al. 2021).
    Example in https://github.com/Project-MONAI/tutorials/blob/master/pathology/multiple_instance_learning/
    panda_mil_train_evaluate_pytorch_gpu.py.
    Args:
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads per layer.
        dim_representation: Dimension of input encoding.
        transformer_dropout: The dropout value of transformer encoder layers.
    """

    def __init__(self, num_layers: int, num_heads: int,
                 dim_representation: int, hidden_dim: int,
                 transformer_dropout: float) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_representation = dim_representation
        self.hidden_dim = hidden_dim
        self.transformer_dropout = transformer_dropout
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.dim_representation,
                                                       nhead=self.num_heads,
                                                       dropout=self.transformer_dropout,
                                                       batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=self.num_layers)
        self.attention = nn.Sequential(nn.Linear(self.dim_representation, self.hidden_dim),
                                       nn.Tanh(), nn.Linear(self.hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input size is L, bag size N, hidden dimension is D, and attention layers K (default K=1).
        """
        x = x.reshape(-1, x.shape[0], x.shape[1])                       # 1 x N X L
        x = self.transformer(x)                                         # 1 x N X L
        a = self.attention(x)                                           # 1 x N X K
        attention_weights = torch.softmax(a, dim=1)                     # 1 x N x K
        pooled_features = torch.sum(x * attention_weights, dim=1)       # K X L
        attention_weights = attention_weights.permute(0, 2, 1)          # 1 x K X N
        return (attention_weights.squeeze(0), pooled_features)          # K X N, K X L
