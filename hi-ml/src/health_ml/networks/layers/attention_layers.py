#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
""" Layers for attention-based multi-instance learning (DeepMIL)
Created using the original DeepMIL paper and code from Ilse et al., 2018
https://github.com/AMLab-Amsterdam/AttentionDeepMIL (MIT License)
"""
from typing import Any, Tuple
from torch import nn, Tensor, transpose, mm
import torch
import torch.nn.functional as F


class MeanPoolingLayer(nn.Module):
    """Mean pooling returns uniform weights and the average feature vector over the first axis"""

    # args/kwargs added here for compatibility with parametrised pooling modules
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

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
