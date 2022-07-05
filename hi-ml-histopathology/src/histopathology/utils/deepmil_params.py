
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import param


class EncoderParams(param.Parameterized):
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
    pool_type: str = param.String(doc="Name of the pooling layer class to use.")
    pool_hidden_dim: int = param.Integer(128, doc="If pooling has a learnable part, this defines the number of the\
        hidden dimensions.")
    pool_out_dim: int = param.Integer(1, doc="Dimension of the pooled representation.")
    num_transformer_pool_layers: int = param.Integer(4, doc="If transformer pooling is chosen, this defines the number\
         of encoding layers.")
    num_transformer_pool_heads: int = param.Integer(4, doc="If transformer pooling is chosen, this defines the number\
         of attention heads.")
