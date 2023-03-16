#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import numpy as np
import torch

from monai.transforms import Compose
from pathlib import Path
from pl_bolts.models.self_supervised import SimCLR
from timm.models import swin_tiny_patch4_window7_224
from timm.models.swin_transformer import SwinTransformer
from torch import Tensor as T, nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import ResNet
from typing import Callable, Optional, Sequence, Tuple

from health_cpath.utils.layer_utils import get_imagenet_preprocessing, load_weights_to_model, setup_feature_extractor


class TileEncoder(nn.Module):
    """Base tile encoder class for use in dataset transforms or as part of a bigger model"""

    def __init__(
        self,
        tile_size: int = 0,
        n_channels: int = 3,
        input_dim: Optional[Sequence[int]] = None,
        checkpoint_activations: bool = False,
    ) -> None:
        """The `TileEncoder` constructor should be called after setting any attributes needed in
        `_get_preprocessing()` or `_get_encoder()`.

        :param tile_size: Tile width/height, in pixels.
        :param n_channels: Number of channels in the tile (default=3).
        :param input_dim: Input shape, to override default of `(n_channels, tile_size, tile_size)`.
        :param checkpoint_activations: Whether to checkpoint activations during forward pass. This can be used to trade
            compute for memory. Note that this is only supported for some encoders.
        """
        super().__init__()
        if input_dim is None:
            if tile_size == 0:
                raise ValueError("Either input_dim or tile_size must be specified")
            input_dim = (n_channels, tile_size, tile_size)
        self.input_dim = tuple(input_dim)

        self.preprocessing_fn = self._get_preprocessing()
        self.feature_extractor_fn, self.num_encoding = self._get_encoder()
        self.checkpoint_activations = checkpoint_activations

    def _get_preprocessing(self) -> Callable:
        return Compose([])

    def _get_encoder(self) -> Tuple[Callable, int]:
        raise NotImplementedError

    def _custom_forward(self, images: torch.Tensor) -> torch.Tensor:
        """A custom forward pass that uses checkpointing to save memory."""
        raise NotImplementedError("Checkpointing is not supported for this encoder")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        prep_images = self.preprocessing_fn(images)
        if self.checkpoint_activations:
            return self._custom_forward(prep_images)
        return self.feature_extractor_fn(prep_images)


class IdentityEncoder(TileEncoder):
    """Dummy encoder that just flattens the input"""

    def _get_encoder(self) -> Tuple[Callable, int]:
        return nn.Flatten(), np.prod(self.input_dim)


class ImageNetEncoder(TileEncoder):
    """Feature extractor pretrained for classification on ImageNet"""

    def __init__(
        self,
        feature_extraction_model: Callable[..., nn.Module],
        tile_size: int,
        n_channels: int = 3,
        apply_imagenet_preprocessing: bool = True,
        checkpoint_activations: bool = False,
    ) -> None:
        """
        :param feature_extraction_model: A function accepting a `pretrained` keyword argument that
        returns a classifier pretrained on ImageNet, such as the ones from `torchvision.models.*`.
        :param tile_size: Tile width/height, in pixels.
        :param n_channels: Number of channels in the tile (default=3).
        :param apply_imagenet_preprocessing: Whether to apply ImageNet preprocessing to the input.
        :param checkpoint_activations: Whether to checkpoint activations during forward pass. This can be used to trade
            compute for memory. Note that this is only supported for some encoders.
        """
        self.create_feature_extractor_fn = feature_extraction_model
        self.apply_imagenet_preprocessing = apply_imagenet_preprocessing
        super().__init__(tile_size=tile_size, n_channels=n_channels, checkpoint_activations=checkpoint_activations)

    def _get_preprocessing(self) -> Callable:
        base_preprocessing = super()._get_preprocessing()
        if self.apply_imagenet_preprocessing:
            return Compose([get_imagenet_preprocessing(), base_preprocessing]).flatten()
        return base_preprocessing

    def _get_encoder(self) -> Tuple[torch.nn.Module, int]:
        pretrained_model = self.create_feature_extractor_fn(pretrained=True)
        return setup_feature_extractor(pretrained_model, self.input_dim)


class ResNetCheckpointingMixin:
    """Mixin class for checkpointing activations in ResNet-based encoders."""

    def __init__(
        self,
        feature_extractor_fn: ResNet,
        bn_momentum: Optional[float] = None,
        checkpoint_segments_size: int = 2,
        checkpoint_activations: bool = False,
    ) -> None:
        """
        :param feature_extractor_fn: A ResNet model.
        :param bn_momentum: An optional momentum value to use for bn layers statistics updates when
            checkpoint_activations is True. If None, sqrt of the default momentum is used.
        :param checkpoint_segments_size: If checkpointing, the size of checkpointed segments in sequential layers
            (default=2).
        :param checkpoint_activations: Whether to checkpoint activations during forward pass. This can be used to trade
            compute for memory. Note that this is only supported for some encoders.
        """
        assert isinstance(feature_extractor_fn, ResNet), "Expected ResNet model for feature_extractor_fn argument."
        self.feature_extractor_fn = feature_extractor_fn
        self.checkpoint_segments_size = checkpoint_segments_size
        self.bn_momentum = bn_momentum
        if checkpoint_activations:
            self._set_batch_norm_momentum()

    def _set_batch_norm_momentum(self) -> None:
        """Set the momentum of batch norm layers in the ResNet model to avoid running statistics from going out of
        sync due to activations checkpointing. The forward pass is applied twice which results in double updates of
        these statistics. We can workaround that by using sqrt of default momentum.
        """
        if self.bn_momentum is not None:
            _momentum = self.bn_momentum
        else:
            _momentum = math.sqrt(self.feature_extractor_fn.bn1.momentum)
            self.bn_momentum = _momentum

        # Set momentum for the first batch norm layer
        self.feature_extractor_fn.bn1.momentum = _momentum

        def _set_bn_momentum(layer_block: nn.Sequential) -> None:
            for sub_layer in layer_block:
                for key, layer in sub_layer._modules.items():
                    if "bn" in key:
                        assert isinstance(layer, nn.BatchNorm2d), "Expected BatchNorm2d layer."
                        layer.momentum = _momentum

        # Fetch all nested batch norm layers and set momentum
        _set_bn_momentum(self.feature_extractor_fn.layer1)
        _set_bn_momentum(self.feature_extractor_fn.layer2)
        _set_bn_momentum(self.feature_extractor_fn.layer3)
        _set_bn_momentum(self.feature_extractor_fn.layer4)

    def _custom_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Custom forward pass that uses activation checkpointing to save memory."""
        segments = self.checkpoint_segments_size
        first_layers = [
            self.feature_extractor_fn.conv1,
            self.feature_extractor_fn.bn1,
            self.feature_extractor_fn.relu,
            self.feature_extractor_fn.maxpool,
        ]
        images = checkpoint_sequential(first_layers, segments, images)
        images = checkpoint_sequential(self.feature_extractor_fn.layer1, segments, images)
        images = checkpoint_sequential(self.feature_extractor_fn.layer2, segments, images)
        images = checkpoint_sequential(self.feature_extractor_fn.layer3, segments, images)
        images = checkpoint_sequential(self.feature_extractor_fn.layer4, segments, images)

        images = checkpoint(self.feature_extractor_fn.avgpool, images)
        images = torch.flatten(images, 1)
        images = checkpoint(self.feature_extractor_fn.fc, images)
        return images


class Resnet18(ResNetCheckpointingMixin, ImageNetEncoder):
    """ResNet18 encoder with imagenet preprocessing."""

    def __init__(
        self,
        tile_size: int,
        n_channels: int = 3,
        checkpoint_activations: bool = False,
        checkpoint_segments_size: int = 2,
        bn_momentum: Optional[float] = None,
    ) -> None:
        """
        :param tile_size: The size of the input tiles.
        :param n_channels: The number of channels in the input tiles.
        :param checkpoint_activations: Whether to checkpoint activations during forward pass.
        :param checkpoint_segments_size: The size of checkpointed segments in sequential layers.
        :param bn_momentum: An optional momentum value to use for bn layers statistics updates when
            checkpoint_activations is True. If None, sqrt of the default momentum is used.
        """
        ImageNetEncoder.__init__(
            self,
            resnet18,
            tile_size,
            n_channels,
            apply_imagenet_preprocessing=True,
            checkpoint_activations=checkpoint_activations,
        )
        ResNetCheckpointingMixin.__init__(
            self, self.feature_extractor_fn, bn_momentum, checkpoint_segments_size, checkpoint_activations
        )


class Resnet18_NoPreproc(ResNetCheckpointingMixin, ImageNetEncoder):
    """ResNet18 encoder without imagenet preprocessing."""

    def __init__(
        self,
        tile_size: int,
        n_channels: int = 3,
        checkpoint_activations: bool = False,
        checkpoint_segments_size: int = 2,
        bn_momentum: Optional[float] = None,
    ) -> None:
        """
        :param tile_size: The size of the input tiles.
        :param n_channels: The number of channels in the input tiles.
        :param checkpoint_activations: Whether to checkpoint activations during forward pass.
        :param checkpoint_segments_size: The size of checkpointed segments in sequential layers.
        :param bn_momentum: An optional momentum value to use for bn layers statistics updates when
            checkpoint_activations is True. If None, sqrt of the default momentum is used.
        """
        ImageNetEncoder.__init__(
            self,
            resnet18,
            tile_size,
            n_channels,
            apply_imagenet_preprocessing=False,
            checkpoint_activations=checkpoint_activations,
        )
        ResNetCheckpointingMixin.__init__(
            self, self.feature_extractor_fn, bn_momentum, checkpoint_segments_size, checkpoint_activations
        )


class Resnet50(ResNetCheckpointingMixin, ImageNetEncoder):
    def __init__(
        self,
        tile_size: int,
        n_channels: int = 3,
        checkpoint_activations: bool = False,
        checkpoint_segments_size: int = 2,
        bn_momentum: Optional[float] = None,
    ) -> None:
        """
        :param tile_size: The size of the input tiles.
        :param n_channels: The number of channels in the input tiles.
        :param checkpoint_activations: Whether to checkpoint activations during forward pass.
        :param checkpoint_segments_size: The size of checkpointed segments in sequential layers.
        :param bn_momentum: An optional momentum value to use for bn layers statistics updates when
            checkpoint_activations is True. If None, sqrt of the default momentum is used.
        """
        ImageNetEncoder.__init__(
            self,
            resnet50,
            tile_size,
            n_channels,
            apply_imagenet_preprocessing=True,
            checkpoint_activations=checkpoint_activations,
        )
        ResNetCheckpointingMixin.__init__(
            self, self.feature_extractor_fn, bn_momentum, checkpoint_segments_size, checkpoint_activations
        )


class Resnet50_NoPreproc(ResNetCheckpointingMixin, ImageNetEncoder):
    def __init__(
        self,
        tile_size: int,
        n_channels: int = 3,
        checkpoint_activations: bool = False,
        checkpoint_segments_size: int = 2,
        bn_momentum: Optional[float] = None,
    ) -> None:
        """
        :param tile_size: The size of the input tiles.
        :param n_channels: The number of channels in the input tiles.
        :param checkpoint_activations: Whether to checkpoint activations during forward pass.
        :param checkpoint_segments_size: The size of checkpointed segments in sequential layers.
        :param bn_momentum: An optional momentum value to use for bn layers statistics updates when
            checkpoint_activations is True. If None, sqrt of the default momentum is used.
        """
        ImageNetEncoder.__init__(
            self,
            resnet50,
            tile_size,
            n_channels,
            apply_imagenet_preprocessing=False,
            checkpoint_activations=checkpoint_activations,
        )
        ResNetCheckpointingMixin.__init__(
            self, self.feature_extractor_fn, bn_momentum, checkpoint_segments_size, checkpoint_activations
        )


class SwinTransformerCheckpointingMixin:
    """Mixin class for checkpointing activations in SwinTransformer-based encoders."""

    def __init__(
        self,
        feature_extractor_fn: ResNet,
        checkpoint_segments_size: int = 2,
    ) -> None:
        """
        :param feature_extractor_fn: A ResNet model.
        :param checkpoint_segments_size: If checkpointing, the size of checkpointed segments in sequential layers
            (default=2).
        """
        assert isinstance(feature_extractor_fn, SwinTransformer), "Expected ResNet model"
        self.feature_extractor_fn = feature_extractor_fn
        self.checkpoint_segments_size = checkpoint_segments_size

    def _custom_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Custom forward pass that uses activation checkpointing to save memory."""
        # patch embedding checkpointing
        images = checkpoint(self.feature_extractor_fn.patch_embed.proj, images)
        images = images.flatten(2).transpose(1, 2)  # BCHW -> BNC
        images = checkpoint(self.feature_extractor_fn.patch_embed.norm, images)
        # do not checkpoint dropout
        images = self.feature_extractor_fn.pos_drop(images)
        # sequential layers checkpointing
        assert isinstance(self.feature_extractor_fn.layers, nn.Sequential), "Expected nn.Sequential for layers."
        images = checkpoint_sequential(self.feature_extractor_fn.layers, self.checkpoint_segments_size, images)
        images = checkpoint(self.feature_extractor_fn.norm, images)
        # AvgPool the output of the last stage to get the feature maps
        images = images.mean(dim=1)
        return images


class SwinTransformer_NoPreproc(SwinTransformerCheckpointingMixin, ImageNetEncoder):
    """Swin Transformer encoder pretrained on ImageNet. This uses the `swin_tiny_patch4_window7_224` model which is a
    tiny version of the Swin Transformer model with a patch size of 4, a window size of 7 and input image size 224."""

    def __init__(
        self,
        tile_size: int,
        n_channels: int = 3,
        checkpoint_activations: bool = False,
        checkpoint_segments_size: int = 2
    ) -> None:
        """
        :param tile_size: The size of the input tiles.
        :param n_channels: The number of channels in the input tiles.
        :param checkpoint_activations: Whether to checkpoint activations during forward pass.
        :param checkpoint_segments_size: The size of checkpointed segments in sequential layers.
        """
        ImageNetEncoder.__init__(
            self,
            swin_tiny_patch4_window7_224,
            tile_size,
            n_channels,
            apply_imagenet_preprocessing=False,
            checkpoint_activations=checkpoint_activations,
        )
        SwinTransformerCheckpointingMixin.__init__(self, self.feature_extractor_fn, checkpoint_segments_size)

    def _get_encoder(self) -> Tuple[torch.nn.Module, int]:
        pretrained_model = self.create_feature_extractor_fn(pretrained=True, num_classes=0)
        return pretrained_model, pretrained_model.num_features  # type: ignore


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
