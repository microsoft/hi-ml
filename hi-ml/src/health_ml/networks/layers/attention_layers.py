#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
""" Layers for attention-based multi-instance learning (DeepMIL)
Created using the original DeepMIL paper and code from Ilse et al., 2018
https://github.com/AMLab-Amsterdam/AttentionDeepMIL (MIT License)
"""
from typing import Callable, Tuple, Optional
from torch import nn, Tensor, transpose, mm
import torch
import torch.nn.functional as F
from torch.nn import Module, MultiheadAttention, Dropout, Linear, LayerNorm


class MeanPoolingLayer(nn.Module):
    """Mean pooling returns uniform weights and the average feature vector over the first axis"""

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        num_instances = features.shape[0]
        A = torch.full((1, num_instances), 1. / num_instances)
        M = features.mean(dim=0)
        M = M.view(1, -1)
        return (A, M)


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
        H = features.view(-1, self.input_dims)      # N x L
        A = self.attention(H)                       # N x K
        A = transpose(A, 1, 0)                      # K x N
        A = F.softmax(A, dim=1)                     # Softmax over N : K x N
        M = mm(A, H)                                # Matrix multiplication : K x L
        return(A, M)


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
        H = features.view(-1, self.input_dims)      # N x L
        A_V = self.attention_V(H)                   # N x D
        A_U = self.attention_U(H)                   # N x D
        A = self.attention_weights(A_V * A_U)       # Element-wise multiplication : N x K
        A = transpose(A, 1, 0)                      # K x N
        A = F.softmax(A, dim=1)                     # Softmax over N : K x N
        M = mm(A, H)                                # Matrix multiplication : K x L
        return(A, M)


class CustomTransformerEncoderLayer(Module):
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
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Callable = F.relu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)  # type: ignore
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)  # type: ignore

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # type: ignore
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # type: ignore
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation

    def forward(self, src: torch.Tensor,
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

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, a = self.self_attn(x, x, x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=True)  # Just because of this flag I had to copy all of the code...
        x = x[0]
        return self.dropout1(x), a

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerPooling(Module):
    """Create a Transformer encoder module consisting of multiple Transformer encoder layers.

    We use a additional classification token (cls token) for pooling like seen in ViT/Bert. First, the cls token is
    appended to the list of tiles encodings. Second, we perform self-attention between all tile encodings and the cls
    token. Last, we extract the cls token and use it for classification.

    Args:
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads per layer.
        dim_representation: Dimension of input encoding.
    """
    def __init__(self, num_layers: int, num_heads: int, dim_representation: int) -> None:
        super(TransformerPooling, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_representation = dim_representation

        self.cls_token = nn.Parameter(torch.zeros(dim_representation))

        self.transformer_encoder_layers = []
        for _ in range(self.num_layers):
            self.transformer_encoder_layers.append(
                CustomTransformerEncoderLayer(self.dim_representation,
                                              self.num_heads,
                                              dim_feedforward=self.dim_representation,
                                              dropout=0.1,
                                              activation=F.gelu,
                                              batch_first=True))
        self.transformer_encoder_layers = torch.nn.ModuleList(self.transformer_encoder_layers)  # type: ignore

    def forward(self, features: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_token = (torch.ones(1, self.dim_representation, device=features.device) * self.cls_token)

        # Append cls token
        H = torch.vstack([cls_token, features]).unsqueeze(0)

        for i in range(self.num_layers):
            H, A = self.transformer_encoder_layers[i](H)

        # Extract cls token
        M = H[:, 0]

        # Get attention weights with respect to the cls token, without the element where it attends to itself
        self_attention_cls_token = A[0, 0, 0]
        A = A[:, 0, 1:]

        # We want A to sum to one, simple hack: add self_attention_cls_token/num_tiles to each element
        A += self_attention_cls_token / A.shape[-1]  # type: ignore

        return (A, M)
