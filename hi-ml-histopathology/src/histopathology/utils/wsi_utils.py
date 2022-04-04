import torch
import numpy as np

from typing import List
from histopathology.utils.naming import SlideKey
from torch.utils.data.dataloader import default_collate


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
