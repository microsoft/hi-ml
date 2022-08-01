import torch
import numpy as np
import logging

from typing import Any, List
from health_cpath.utils.naming import SlideKey
from health_ml.utils.bag_utils import multibag_collate
from monai.utils import WSIPatchKeys
from monai.data import MetaTensor

slide_metadata_keys = [
    SlideKey.IMAGE_PATH,
    SlideKey.LABEL,
    SlideKey.MASK,
    SlideKey.METADATA,
    SlideKey.SLIDE_ID,
    SlideKey.MASK_PATH,
    WSIPatchKeys.COUNT,
    SlideKey.TILE_SIZE,  # TODO: remove in case we want to allow patches of different sizes from the same slide
    SlideKey.SHAPE,
    SlideKey.OFFSET
]


def array_collate(batch: List) -> Any:
    """
        Combine instances from a list of dicts into a single dict, by stacking arrays along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW. It also convert some values to tensors.
        The list of dicts refers to the list of tiles produced by GridPatch transform applied on a WSI.
    """
    collate_keys = []
    constant_keys = slide_metadata_keys
    for key in batch[0][0].keys():
        if key not in slide_metadata_keys:
            if isinstance(batch[0][0][key], np.ndarray) or isinstance(batch[0][0][key], MetaTensor):
                collate_keys.append(key)
            else:
                logging.warning("Only np.ndarray and MetaTensors are collated -"
                                f"{key} value will be taken from first patch")
                constant_keys.append(key)
    tensor_keys = collate_keys + [SlideKey.LABEL]

    new_batch: List[dict] = []
    for patch_data in batch:
        # we assume all patches are dictionaries with the same keys
        data = patch_data[0]
        for key in collate_keys:
            if isinstance(data[key], np.ndarray):
                data[key] = np.array([ix[key] for ix in patch_data])
            elif isinstance(data[key], MetaTensor):
                # TODO change how this collation happens if we have list of tensors
                data[key] = np.array([ix[key].as_tensor().numpy() for ix in patch_data])
        for key in tensor_keys:
            data[key] = torch.tensor(data[key])
        new_batch.append(data)
        batch = new_batch
    return multibag_collate(batch)
