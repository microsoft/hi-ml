import torch
import numpy as np

from typing import List, Dict
from histopathology.utils.naming import SlideKey, ResultsKey
from torch.utils.data.dataloader import default_collate
from monai.transforms.transform import Transform
from monai.utils import convert_to_dst_type, convert_data_type
from monai.config.type_definitions import NdarrayOrTensor
from preprocessing.tiling import tile_array_2d, assemble_tiles_2d, generate_tiles


def image_collate(batch: List[dict]) -> List[dict]:
    """
        Combine instances from a list of dicts into a single dict, by stacking them along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW.
        The list of dicts refers to the the list of tiles produced by the TileOnGridd transform applied on a WSI.
    """

    for i, item in enumerate(batch):
        data = item[0]
        data[SlideKey.IMAGE] = torch.tensor(np.array([ix[SlideKey.IMAGE] for ix in item]))
        batch[i] = data
    return default_collate(batch)


def GridTiling(Transform):
    def __init__(
        self,
        tile_size: int = 256,
        background_val: int = 255,
    ) -> None:
        self.tile_size = tile_size
        self.background_val = background_val


    def __call__(self, image: NdarrayOrTensor) -> Dict(NdarrayOrTensor, NdarrayOrTensor):
        img_np: np.ndarray
        img_np, *_ = convert_data_type(image, np.ndarray)  # type: ignore
        generate_tiles(image_np, tile_size=self.tile_size, 
        foreground_threshold=self.background_val, occupancy_threshold: self.occupancy_threshols)
        tiles, coords = tile_array_2d(img_np, self.tile_size, constant_value=self.background_val)
        array, offset = assemble_tiles_2d(tiles, coords)
        image, *_ = convert_to_dst_type(src=array, dst=image, dtype=image.dtype)
        return {SlideKey.IMAGE: image, ResultsKey.TILE_ID: coords}
