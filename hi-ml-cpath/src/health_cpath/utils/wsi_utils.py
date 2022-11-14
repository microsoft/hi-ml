import torch
import param
import numpy as np

from typing import Any, List, Optional
from health_cpath.utils.naming import SlideKey
from health_ml.utils.bag_utils import multibag_collate


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


class TileOnTheFlyParams(param.Parameterized):
    """
        Parameters for the TileOnTheFly transform.
    """

    tile_size: int = param.Integer(default=224, bounds=(1, None), doc="The size of the tile, Default: 224")
    overlap: int = param.Number(default=0, bounds=(0.0, 1.0),
                                doc="The amount of overlap of neighboring patches in each dimension (a value between "
                                   "0.0 and 1.0). Defaults to 0.0.")
    sort_fn: Optional[str] = param.String(default=None,
                                          doc="When bag_size is fixed, it determines whether to keep tiles with "
                                              "highest intensity values (`'max'`), lowest values (`'min'`) that assumes"
                                              "background is high values, or in their default order (`None`). "
                                              "Default to None.")
    pad_mode: Optional[str] = param.String(default='constant',
                                           doc="The mode of padding, defaults to 'constant'. Refer to NumpyPadMode and "
                                               "PytorchPadMode. If None, no padding will be applied.")
    intensity_threshold: Optional[float] = param.Number(default=None,


    def _get_tile_on_the_fly_transform(self, stage: ModelKey, image_key: str, num_tiles: int) -> Callable:
        if stage == ModelKey.TRAIN:
            return RandGridPatchd(
                keys=[image_key],
                patch_size=(self.tile_size, self.tile_size),
                num_patches=num_tiles,
                sort_fn=self.filter_mode,
                pad_mode=self.pad_mode,  # type: ignore
                constant_values=self.background_val,
                overlap=self.overlap,  # type: ignore
                threshold=self.background_val,
            )
        else:
            return GridPatchd(
                keys=[image_key],
                tile_count=self.bag_sizes[stage],
                patch_size=self.tile_size,
                tile_size=self.tile_size,
                step=self.step,
                random_offset=self.random_offset if stage == ModelKey.TRAIN else False,
                pad_full=self.pad_full,
                background_val=self.background_val,
                filter_mode=self.filter_mode,
                return_list_of_dicts=True,
            )
