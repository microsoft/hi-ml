
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import param
from torch import nn
from pathlib import Path
from typing import Optional, Tuple
from torchvision.models.resnet import resnet18, resnet50

from histopathology.models.encoders import (
    HistoSSLEncoder, IdentityEncoder, ImageNetEncoder, ImageNetEncoder_Resnet50, ImageNetSimCLREncoder,
    SSLEncoder, TileEncoder)
from histopathology.utils.download_utils import get_checkpoint_downloader

from health_ml.networks.layers.attention_layers import (AttentionLayer, GatedAttentionLayer, MaxPoolingLayer,
                                                        MeanPoolingLayer, TransformerPooling,
                                                        TransformerPoolingBenchmark)


class EncoderParams(param.Parameterized):
    """Parameters class to group all encoder specific attributes for deepmil module. """

    encoder_type: str = param.String(doc="Name of the encoder class to use.")
    tile_size: int = param.Integer(224, bounds=(1, None), doc="Tile width/height, in pixels.")
    n_channels: int = param.Integer(3, bounds=(1, None), doc="Number of channels in the tile.")
    is_finetune: bool = param.Boolean(False, doc="If True, fine-tune the encoder during training. If False (default), "
                                                 "keep the encoder frozen.")
    is_caching: bool = param.Boolean(False, doc="If True, cache the encoded tile features "
                                                "(disables random subsampling of tiles). "
                                                "If False (default), load the tiles without caching "
                                                "(enables random subsampling of tiles).")
    encoding_chunk_size: int = param.Integer(0, doc="If > 0 performs encoding in chunks, by enconding_chunk_size tiles "
                                                    "per chunk")


class PoolingParams(param.Parameterized):
    """Parameters class to group all pooling specific attributes for deepmil module. """

    pool_type: str = param.String(doc="Name of the pooling layer class to use.")
    pool_hidden_dim: int = param.Integer(128, doc="If pooling has a learnable part, this defines the number of the\
        hidden dimensions.")
    pool_out_dim: int = param.Integer(1, doc="Dimension of the pooled representation.")
    num_transformer_pool_layers: int = param.Integer(4, doc="If transformer pooling is chosen, this defines the number\
         of encoding layers.")
    num_transformer_pool_heads: int = param.Integer(4, doc="If transformer pooling is chosen, this defines the number\
         of attention heads.")


def get_encoder(
    ckpt_run_id: Optional[str], outputs_folder: Optional[Path], encoder_params: EncoderParams
) -> TileEncoder:
    """Given the encoder parameters, returns the encoder object.

    :param ckpt_run_id: The AML run id for SSL checkpoint download.
    :param outputs_folder: The output folder where SSL checkpoint should be saved.
    :param encoder_params: The encoder arguments that define the encoder class object depending on the encoder type.
    :raises ValueError: If the encoder type is not supported.
    :return: A TileEncoder instance for deepmil module.
    """
    encoder: TileEncoder
    if encoder_params.encoder_type == ImageNetEncoder.__name__:
        encoder = ImageNetEncoder(
            feature_extraction_model=resnet18,
            tile_size=encoder_params.tile_size,
            n_channels=encoder_params.n_channels,
        )
    elif encoder_params.encoder_type == ImageNetEncoder_Resnet50.__name__:
        # Myronenko et al. 2021 uses Resnet50 CNN encoder
        encoder = ImageNetEncoder_Resnet50(
            feature_extraction_model=resnet50,
            tile_size=encoder_params.tile_size,
            n_channels=encoder_params.n_channels,
        )

    elif encoder_params.encoder_type == ImageNetSimCLREncoder.__name__:
        encoder = ImageNetSimCLREncoder(
            tile_size=encoder_params.tile_size, n_channels=encoder_params.n_channels
        )

    elif encoder_params.encoder_type == HistoSSLEncoder.__name__:
        encoder = HistoSSLEncoder(
            tile_size=encoder_params.tile_size, n_channels=encoder_params.n_channels
        )

    elif encoder_params.encoder_type == SSLEncoder.__name__:
        assert ckpt_run_id and outputs_folder, "SSLEncoder requires ckpt_run_id and outputs_folder"
        downloader = get_checkpoint_downloader(ckpt_run_id, outputs_folder)
        encoder = SSLEncoder(
            pl_checkpoint_path=downloader.local_checkpoint_path,
            tile_size=encoder_params.tile_size,
            n_channels=encoder_params.n_channels,
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_params.encoder_type}")

    if encoder_params.is_finetune:
        for params in encoder.parameters():
            params.requires_grad = True
    else:
        encoder.eval()
    if encoder_params.is_caching:
        # Encoding is done in the datamodule, so here we provide instead a dummy
        # no-op IdentityEncoder to be used inside the model
        encoder = IdentityEncoder(input_dim=(encoder.num_encoding,))
    return encoder


def get_pooling_layer(pooling_params: PoolingParams, num_encoding: int) -> Tuple[nn.Module, int]:
    """Given the pooling parameters, returns the pooling layer object.

    :param pooling_params: Pooling parameters that define the pooling layer class object depending on the pooling type.
    :param num_encoding: The embedding dimension of the encoder.
    :raises ValueError: If the pooling type is not supported.
    :return: A tuple of (pooling layer, pooling output dimension) for deepmil module.
    """
    pooling_layer: nn.Module
    if pooling_params.pool_type == AttentionLayer.__name__:
        pooling_layer = AttentionLayer(num_encoding, pooling_params.pool_hidden_dim, pooling_params.pool_out_dim)
    elif pooling_params.pool_type == GatedAttentionLayer.__name__:
        pooling_layer = GatedAttentionLayer(num_encoding,
                                            pooling_params.pool_hidden_dim,
                                            pooling_params.pool_out_dim)
    elif pooling_params.pool_type == MeanPoolingLayer.__name__:
        pooling_layer = MeanPoolingLayer()
    elif pooling_params.pool_type == MaxPoolingLayer.__name__:
        pooling_layer = MaxPoolingLayer()
    elif pooling_params.pool_type == TransformerPooling.__name__:
        pooling_layer = TransformerPooling(pooling_params.num_transformer_pool_layers,
                                           pooling_params.num_transformer_pool_heads,
                                           num_encoding)
        pooling_params.pool_out_dim = 1  # currently this is hardcoded in forward of the TransformerPooling
    elif pooling_params.pool_type == TransformerPoolingBenchmark.__name__:
        pooling_layer = TransformerPoolingBenchmark(pooling_params.num_transformer_pool_layers,
                                                    pooling_params.num_transformer_pool_heads,
                                                    num_encoding,
                                                    pooling_params.pool_hidden_dim)
        pooling_params.pool_out_dim = 1  # currently this is hardcoded in forward of the TransformerPooling
    else:
        raise ValueError(f"Unsupported pooling type: {pooling_params.pooling_type}")
    num_features = num_encoding * pooling_params.pool_out_dim
    return pooling_layer, num_features
