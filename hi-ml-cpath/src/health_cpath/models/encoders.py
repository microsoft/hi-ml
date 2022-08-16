#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from pl_bolts.models.self_supervised import SimCLR
from torch import Tensor as T, nn
from torchvision.models import resnet18, resnet50
from monai.transforms import Compose

from health_cpath.utils.layer_utils import (get_imagenet_preprocessing,
                                            load_weights_to_model,
                                            setup_feature_extractor)


class TileEncoder(nn.Module):
    """Base tile encoder class for use in dataset transforms or as part of a bigger model"""

    def __init__(self, tile_size: int = 0, n_channels: int = 3,
                 input_dim: Optional[Sequence[int]] = None) -> None:
        """The `TileEncoder` constructor should be called after setting any attributes needed in
        `_get_preprocessing()` or `_get_encoder()`.

        :param tile_size: Tile width/height, in pixels.
        :param n_channels: Number of channels in the tile (default=3).
        :param input_dim: Input shape, to override default of `(n_channels, tile_size, tile_size)`.
        """
        super().__init__()
        if input_dim is None:
            if tile_size == 0:
                raise ValueError("Either input_dim or tile_size must be specified")
            input_dim = (n_channels, tile_size, tile_size)
        self.input_dim = tuple(input_dim)

        self.preprocessing_fn = self._get_preprocessing()
        self.feature_extractor_fn, self.num_encoding = self._get_encoder()

    def _get_preprocessing(self) -> Callable:
        return Compose([])

    def _get_encoder(self) -> Tuple[Callable, int]:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        prep_images = self.preprocessing_fn(images)
        return self.feature_extractor_fn(prep_images)


class IdentityEncoder(TileEncoder):
    """Dummy encoder that just flattens the input"""

    def _get_encoder(self) -> Tuple[Callable, int]:
        return nn.Flatten(), np.prod(self.input_dim)


class ImageNetEncoder(TileEncoder):
    """Feature extractor pretrained for classification on ImageNet"""

    def __init__(self, feature_extraction_model: Callable[..., nn.Module],
                 tile_size: int, n_channels: int = 3, apply_imagenet_preprocessing: bool = True) -> None:
        """
        :param feature_extraction_model: A function accepting a `pretrained` keyword argument that
        returns a classifier pretrained on ImageNet, such as the ones from `torchvision.models.*`.
        :param tile_size: Tile width/height, in pixels.
        :param n_channels: Number of channels in the tile (default=3).
        :param apply_imagenet_preprocessing: Whether to apply ImageNet preprocessing to the input.
        """
        self.create_feature_extractor_fn = feature_extraction_model
        self.apply_imagenet_preprocessing = apply_imagenet_preprocessing
        super().__init__(tile_size=tile_size, n_channels=n_channels)

    def _get_preprocessing(self) -> Callable:
        base_preprocessing = super()._get_preprocessing()
        if self.apply_imagenet_preprocessing:
            return Compose([get_imagenet_preprocessing(), base_preprocessing]).flatten()
        return base_preprocessing

    def _get_encoder(self) -> Tuple[torch.nn.Module, int]:
        pretrained_model = self.create_feature_extractor_fn(pretrained=True)
        return setup_feature_extractor(pretrained_model, self.input_dim)


class Resnet18(ImageNetEncoder):
    def __init__(self, tile_size: int, n_channels: int = 3) -> None:
        super().__init__(resnet18, tile_size, n_channels, apply_imagenet_preprocessing=True)


class Resnet18_NoPreproc(ImageNetEncoder):
    def __init__(self, tile_size: int, n_channels: int = 3) -> None:
        super().__init__(resnet18, tile_size, n_channels, apply_imagenet_preprocessing=False)


class Resnet50(ImageNetEncoder):
    def __init__(self, tile_size: int, n_channels: int = 3) -> None:
        super().__init__(resnet50, tile_size, n_channels, apply_imagenet_preprocessing=True)


class Resnet50_NoPreproc(ImageNetEncoder):
    def __init__(self, tile_size: int, n_channels: int = 3) -> None:
        super().__init__(resnet50, tile_size, n_channels, apply_imagenet_preprocessing=False)


class ImageNetSimCLREncoder(TileEncoder):
    """SimCLR encoder pretrained on ImageNet"""

    WEIGHTS_URL = ("https://pl-bolts-weights.s3.us-east-2.amazonaws.com/"
                   "simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt")
    EMBEDDING_DIM = 2048

    def _get_preprocessing(self) -> Callable:
        return get_imagenet_preprocessing()

    def _get_encoder(self) -> Tuple[SimCLR, int]:
        simclr = SimCLR.load_from_checkpoint(self.WEIGHTS_URL, strict=False)
        simclr.freeze()
        return simclr, self.EMBEDDING_DIM


class SSLEncoder(TileEncoder):
    """SSL encoder trained on Azure ML"""

    def __init__(self, pl_checkpoint_path: Path, tile_size: int, n_channels: int = 3) -> None:
        """
        :param pl_checkpoint_path: The path of the downloaded checkpoint file.
        :param tile_size: Tile width/height, in pixels.
        :param n_channels: Number of channels in the tile (default=3).
        """
        self.pl_checkpoint_path = pl_checkpoint_path
        super().__init__(tile_size=tile_size, n_channels=n_channels)

    def _get_encoder(self) -> Tuple[torch.nn.Module, int]:
        try:
            from SSL.lightning_modules.ssl_classifier_module import SSLClassifier
            from SSL.utils import create_ssl_image_classifier
            from SSL import encoders
        except (ImportError, ModuleNotFoundError):
            raise ValueError("SSL not found. This class can only be used by using hi-ml from the GitHub source")
        model: SSLClassifier = create_ssl_image_classifier(  # type: ignore
            num_classes=1,  # dummy value
            freeze_encoder=True,
            pl_checkpoint_path=str(self.pl_checkpoint_path)
        )
        assert isinstance(model.encoder, encoders.SSLEncoder)
        return setup_feature_extractor(model.encoder.cnn_model, self.input_dim)

    def forward(self, x: T) -> T:
        x = super().forward(x)
        return x[-1] if isinstance(x, list) else x


class HistoSSLEncoder(TileEncoder):
    """HistoSSL encoder pretrained on multiple histological datasets

    Reference:
    - Ciga, Xu, Martel (2021). Self supervised contrastive learning for digital histopathology.
    arXiv:2011.13971
    """

    WEIGHTS_URL = ("https://github.com/ozanciga/self-supervised-histopathology/releases/"
                   "download/tenpercent/tenpercent_resnet18.ckpt")

    def _get_encoder(self) -> Tuple[torch.nn.Module, int]:
        resnet18_model = resnet18(pretrained=False)
        histossl_encoder = load_weights_to_model(self.WEIGHTS_URL, resnet18_model)
        return setup_feature_extractor(histossl_encoder, self.input_dim)
