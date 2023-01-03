import torch
import param
from torch import Tensor
from typing import Any, Callable, List, Optional
from health_cpath.models.transforms import ExtractCoordinatesd
from health_cpath.utils.naming import ModelKey, SlideKey
from health_ml.utils.bag_utils import multibag_collate
from monai.transforms import RandGridPatchd, GridPatchd, SplitDimd


def image_collate(batch: List) -> Any:
    """
        Combine instances from a list of dicts into a single dict, by stacking them along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW.
        The list of dicts refers to the the list of tiles produced by the Rand/GridPatchd transform applied on a WSI.
    """

    for i, item in enumerate(batch):
        # The tiles have been splited into a list of dicts, each dict containing a single tile to be able to apply
        # tile wise transforms. We need to stack them back together.
        data = item[0]
        assert isinstance(data[SlideKey.IMAGE], Tensor), f"Expected torch.Tensor, got {type(data[SlideKey.IMAGE])}"
        data[SlideKey.IMAGE] = torch.stack([ix[SlideKey.IMAGE] for ix in item], dim=0)
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
        doc="The mode of padding, refer to NumpyPadMode and PytorchPadMode. Defaults to None, for no padding.")
    intensity_threshold: float = param.Number(
        default=255.,
        doc="The intensity threshold to filter out tiles based on intensity values. Default to None.")
    background_val: int = param.Integer(
        default=255,
        doc="The intensity value of background. Default to 255.")
    rand_min_offset: int = param.Integer(
        default=0,
        bounds=(0, None),
        doc="The minimum range of sarting position to be selected randomly. This parameter is passed to RandGridPatchd."
            "the random version of RandGridPatchd used at training time. Default to 0.")
    rand_max_offset: int = param.Integer(
        default=None,
        bounds=(0, None),
        doc="The maximum range of sarting position to be selected randomly. This parameter is passed to RandGridPatchd."
            "the random version of RandGridPatchd used at training time. Default to None.")
    inf_offset: Optional[int] = param.Integer(
        default=None,
        doc="The offset to be used for inference sampling. This parameter is passed to GridPatchd. Default to None.")

    @property
    def scaled_threshold(self) -> float:
        """Returns the threshold to be used for filtering out tiles based on intensity values. We need to multiply
        the threshold by the tile size to account for the fact that the intensity is computed on the entire tile"""
        return 0.999 * 3 * self.intensity_threshold * self.tile_size * self.tile_size

    def get_tiling_transform(self, bag_size: int, stage: ModelKey,) -> Callable:
        if stage == ModelKey.TRAIN:
            return RandGridPatchd(
                keys=[SlideKey.IMAGE],
                patch_size=(self.tile_size, self.tile_size),
                min_offset=self.rand_min_offset,
                max_offset=self.rand_max_offset,
                num_patches=bag_size,
                overlap=self.tile_overlap,
                sort_fn=self.tile_sort_fn,
                threshold=self.scaled_threshold,
                pad_mode=self.tile_pad_mode,  # type: ignore
                constant_values=self.background_val,  # this arg is passed to np.pad or torch.pad
            )
        else:
            return GridPatchd(
                keys=[SlideKey.IMAGE],
                patch_size=(self.tile_size, self.tile_size),
                offset=self.inf_offset,  # type: ignore
                num_patches=bag_size,
                overlap=self.tile_overlap,
                sort_fn=self.tile_sort_fn,
                threshold=self.scaled_threshold,
                pad_mode=self.tile_pad_mode,  # type: ignore
                constant_values=self.background_val,  # this arg is passed to np.pad or torch.pad
            )

    def get_split_transform(self) -> Callable:
        """GridPatchd returns stacked tiles (bag_size, C, H, W), however we need to split them into separate
        tiles to be able to apply augmentations on each tile independently.
        """
        return SplitDimd(keys=SlideKey.IMAGE, dim=0, keepdim=False, list_output=True)

    def get_extract_coordinates_transform(self) -> Callable:
        """Extract the coordinates of the tiles returned as meta data by monai transforms to hi-ml-cpath format where
        the coordinates are represented as TileKey.TILE_LEFT, TileKey.TILE_TOP, TileKey.TILE_RIGHT, TileKey.TILE_BOTTOM.
        """
        return ExtractCoordinatesd(image_key=SlideKey.IMAGE, tile_size=self.tile_size)
