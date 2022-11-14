import torch
import param
import numpy as np

from typing import Any, Callable, List, Optional
from health_cpath.utils.naming import ModelKey, SlideKey
from health_ml.utils.bag_utils import multibag_collate

from monai.transforms import RandGridPatchd, GridPatchd


def image_collate(batch: List) -> Any:
    """
        Combine instances from a list of dicts into a single dict, by stacking them along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW.
        The list of dicts refers to the the list of tiles produced by the TileOnGridd transform applied on a WSI.
    """

    for i, item in enumerate(batch):
        data = item[0]
        if isinstance(data[SlideKey.IMAGE], torch.Tensor):
            data[SlideKey.IMAGE] = torch.stack([ix[SlideKey.IMAGE] for ix in item], dim=0)
        else:
            data[SlideKey.IMAGE] = torch.tensor(np.array([ix[SlideKey.IMAGE] for ix in item]))
        data[SlideKey.LABEL] = torch.tensor(data[SlideKey.LABEL])
        batch[i] = data
    return multibag_collate(batch)


class TilingParams(param.Parameterized):
    """Parameters for Tiling On the Fly a WSI using RandGridPatchd and GridPatchd monai transforms"""

    tile_size: int = param.Integer(default=224, bounds=(1, None), doc="The size of the tile, Default: 224")
    tile_overlap: int = param.Number(
        default=0,
        bounds=(0.0, 1.0),
        doc="The amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).")
    tile_sort_fn: Optional[str] = param.String(
        default='min',
        doc="When bag_size is fixed, it determines whether to keep tiles with highest intensity values (`'max'`), "
            "lowest values (`'min'`) that assumes background is high values, or in their default order (`None`). ")
    tile_pad_mode: Optional[str] = param.String(
        default=None,
        doc="The mode of padding, defaults. Refer to NumpyPadMode and PytorchPadMode. "
            "Defaults to None, no padding will be applied.")
    intensity_threshold: Optional[float] = param.Number(
        default=None,
        doc="The intensity threshold to filter out tiles based on intensity values. Default to None.")
    background_intensity: int = param.Integer(
        default=255,
        doc="The intensity value of background. Default to 255.")
    rand_min_offset: int = param.Integer(
        default=0,
        bounds=(0, None),
        doc="The minimum range of sarting position to be selected randomly. This parameter is passed to RandGridPatchd."
            "the random version of GridPatchd used at training time. Default to 0.")
    rand_max_offset: int = param.Integer(
        default=None,
        bounds=(0, None),
        doc="The maximum range of sarting position to be selected randomly. This parameter is passed to RandGridPatchd."
            "the random version of GridPatchd used at training time. Default to None.")

    def get_tiling_transform(self, stage: ModelKey, image_key: str, bag_size: int) -> Callable:
        if stage == ModelKey.TRAIN:
            return RandGridPatchd(
                keys=[image_key],
                patch_size=(self.tile_size, self.tile_size),
                min_offset=self.rand_min_offset,
                max_offset=self.rand_max_offset,
                num_patches=bag_size,
                overlap=self.tile_overlap,
                sort_fn=self.tile_sort_fn,
                threshold=self.intensity_threshold,
                pad_mode=self.tile_pad_mode,
                constant_values=self.background_intensity,  # arg passed to np.pad or torch.pad
            )
        else:
            return GridPatchd(
                keys=[image_key],
                patch_size=(self.tile_size, self.tile_size),
                num_patches=bag_size,
                overlap=self.tile_overlap,
                sort_fn=self.tile_sort_fn,
                threshold=self.intensity_threshold,
                pad_mode=self.tile_pad_mode,
                constant_values=self.background_intensity,  # arg passed to np.pad or torch.pad
            )
