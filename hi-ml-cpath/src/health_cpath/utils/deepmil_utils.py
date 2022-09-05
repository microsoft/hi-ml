#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import param
from torch import nn
from pathlib import Path
from typing import Optional, Tuple
from health_ml.utils.checkpoint_utils import LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, CheckpointDownloader
from health_ml.utils.common_utils import DEFAULT_AML_CHECKPOINT_DIR
from health_cpath.models.encoders import (
    HistoSSLEncoder,
    ImageNetSimCLREncoder,
    SSLEncoder,
    TileEncoder,
    Resnet18,
    Resnet50,
    Resnet18_NoPreproc,
    Resnet50_NoPreproc,
)
from health_ml.networks.layers.attention_layers import (
    AttentionLayer,
    GatedAttentionLayer,
    MaxPoolingLayer,
    MeanPoolingLayer,
    TransformerPooling,
    TransformerPoolingBenchmark,
)


def set_module_gradients_enabled(model: nn.Module, tuning_flag: bool) -> None:
    """Given a model, enable or disable gradients for all parameters.

    :param model: A PyTorch model.
    :param tuning_flag: A boolean indicating whether to enable or disable gradients for the model parameters.
    """
    for params in model.parameters():
        params.requires_grad = tuning_flag


class EncoderParams(param.Parameterized):
    """Parameters class to group all encoder specific attributes for deepmil module. """

    encoder_type: str = param.String(doc="Name of the encoder class to use.")
    tile_size: int = param.Integer(default=224, bounds=(1, None), doc="Tile width/height, in pixels.")
    n_channels: int = param.Integer(default=3, bounds=(1, None), doc="Number of channels in the tile.")
    tune_encoder: bool = param.Boolean(
        False, doc="If True, fine-tune the encoder during training. If False (default), keep the encoder frozen."
    )
    pretrained_encoder = param.Boolean(
        False, doc="If True, transfer weights from the pretrained model (specified in `src_checkpoint`) to the encoder."
        "Else (False), keep the encoder weights as defined by the `encoder_type`."
    )
    is_caching: bool = param.Boolean(
        default=False,
        doc="If True, cache the encoded tile features (disables random subsampling of tiles). If False (default), load "
        "the tiles without caching (enables random subsampling of tiles).",
    )
    encoding_chunk_size: int = param.Integer(
        default=0, doc="If > 0 performs encoding in chunks, by enconding_chunk_size tiles " "per chunk"
    )

    def get_encoder(self, ssl_ckpt_run_id: Optional[str], outputs_folder: Optional[Path]) -> TileEncoder:
        """Given the current encoder parameters, returns the encoder object.

        :param ssl_ckpt_run_id: The AML run id for SSL checkpoint download.
        :param outputs_folder: The output folder where SSL checkpoint should be saved.
        :param encoder_params: The encoder arguments that define the encoder class object depending on the encoder type.
        :raises ValueError: If the encoder type is not supported.
        :return: A TileEncoder instance for deepmil module.
        """
        encoder: TileEncoder
        if self.encoder_type == Resnet18.__name__:
            encoder = Resnet18(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == Resnet18_NoPreproc.__name__:
            encoder = Resnet18_NoPreproc(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == Resnet50.__name__:
            encoder = Resnet50(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == Resnet50_NoPreproc.__name__:
            encoder = Resnet50_NoPreproc(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == ImageNetSimCLREncoder.__name__:
            encoder = ImageNetSimCLREncoder(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == HistoSSLEncoder.__name__:
            encoder = HistoSSLEncoder(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == SSLEncoder.__name__:
            assert ssl_ckpt_run_id and outputs_folder, "SSLEncoder requires ssl_ckpt_run_id and outputs_folder"
            downloader = CheckpointDownloader(
                run_id=ssl_ckpt_run_id,
                download_dir=outputs_folder,
                checkpoint_filename=LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX,
                remote_checkpoint_dir=Path(DEFAULT_AML_CHECKPOINT_DIR),
            )
            encoder = SSLEncoder(
                pl_checkpoint_path=downloader.local_checkpoint_path,
                tile_size=self.tile_size,
                n_channels=self.n_channels,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")
        set_module_gradients_enabled(encoder, tuning_flag=self.tune_encoder)
        return encoder


class PoolingParams(param.Parameterized):
    """Parameters class to group all pooling specific attributes for deepmil module. """

    pool_type: str = param.String(doc="Name of the pooling layer class to use.")
    pool_hidden_dim: int = param.Integer(
        default=128, doc="If pooling has a learnable part, this defines the number of the hidden dimensions.",
    )
    pool_out_dim: int = param.Integer(1, doc="Dimension of the pooled representation.")
    num_transformer_pool_layers: int = param.Integer(
        default=4, doc="If transformer pooling is chosen, this defines the number of encoding layers.",
    )
    num_transformer_pool_heads: int = param.Integer(
        default=4, doc="If transformer pooling is chosen, this defines the number of attention heads.",
    )
    tune_pooling: bool = param.Boolean(
        default=True,
        doc="If True (default), fine-tune the pooling layer during training. If False, keep the pooling layer frozen.",
    )
    pretrained_pooling = param.Boolean(
        default=False,
        doc="If True, transfer weights from the pretrained model (specified in `src_checkpoint`) to the pooling"
        "layer. Else (False), initialize the pooling layer randomly.",
    )
    transformer_dropout: float = param.Number(
        default=0.0,
        doc="If transformer pooling is chosen, this defines the dropout of the tranformer encoder layers.",
    )

    def get_pooling_layer(self, num_encoding: int) -> Tuple[nn.Module, int]:
        """Given the pooling parameters, returns the pooling layer object.

        :param pooling_params: Pooling parameters that define the pooling layer class depending on the pooling type.
        :param num_encoding: The embedding dimension of the encoder.
        :raises ValueError: If the pooling type is not supported.
        :return: A tuple of (pooling layer, pooling output dimension) for deepmil module.
        """
        pooling_layer: nn.Module
        if self.pool_type == AttentionLayer.__name__:
            pooling_layer = AttentionLayer(input_dims=num_encoding,
                                           hidden_dims=self.pool_hidden_dim,
                                           attention_dims=self.pool_out_dim)
        elif self.pool_type == GatedAttentionLayer.__name__:
            pooling_layer = GatedAttentionLayer(input_dims=num_encoding,
                                                hidden_dims=self.pool_hidden_dim,
                                                attention_dims=self.pool_out_dim)
        elif self.pool_type == MeanPoolingLayer.__name__:
            pooling_layer = MeanPoolingLayer()
        elif self.pool_type == MaxPoolingLayer.__name__:
            pooling_layer = MaxPoolingLayer()
        elif self.pool_type == TransformerPooling.__name__:
            pooling_layer = TransformerPooling(
                num_layers=self.num_transformer_pool_layers,
                num_heads=self.num_transformer_pool_heads,
                dim_representation=num_encoding,
                transformer_dropout=self.transformer_dropout)
            self.pool_out_dim = 1  # currently this is hardcoded in forward of the TransformerPooling
        elif self.pool_type == TransformerPoolingBenchmark.__name__:
            pooling_layer = TransformerPoolingBenchmark(
                num_layers=self.num_transformer_pool_layers,
                num_heads=self.num_transformer_pool_heads,
                dim_representation=num_encoding,
                hidden_dim=self.pool_hidden_dim,
                transformer_dropout=self.transformer_dropout)
            self.pool_out_dim = 1  # currently this is hardcoded in forward of the TransformerPooling
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")
        num_features = num_encoding * self.pool_out_dim
        set_module_gradients_enabled(pooling_layer, tuning_flag=self.tune_pooling)
        return pooling_layer, num_features
