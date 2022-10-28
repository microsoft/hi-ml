import torch
import numpy as np

from typing import Any, List
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
