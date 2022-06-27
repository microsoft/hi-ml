import torch
import numpy as np
import logging

from typing import Any, List
from histopathology.utils.naming import SlideKey
from health_ml.utils.bag_utils import multibag_collate
from monai.utils import WSIPatchKeys

slide_metadata_keys = [
    SlideKey.IMAGE_PATH,
    SlideKey.LABEL,
    SlideKey.MASK,
    SlideKey.METADATA,
    SlideKey.SLIDE_ID,
    SlideKey.MASK_PATH,
    WSIPatchKeys.COUNT,
    SlideKey.PATCH_SIZE,  # TODO: remove in case we want to allow patches of different sizes from the same slide
    SlideKey.SHAPE,
    SlideKey.OFFSET
]


def check_patch_location_format(batch):
    """
    check locations returned by transform have expected size [z, y, x]
    """
    faulty_slides_idx = []
    for slide_data in batch:
        for patch in slide_data:
            location = patch[SlideKey.PATCH_LOCATION]
            if not isinstance(location[0], np.uint8):
                # we assume the location is 2d [y, x] but MONAI sometimes returns [[0], [0]] instead
                faulty_slides_idx.append(patch[SlideKey.SLIDE_ID])
                break
    n = len(faulty_slides_idx)
    if n > 0:
        logging.warning(f'{n} slides will be skipped because something was wrong in the patch location')
    return faulty_slides_idx


def array_collate(batch: List) -> Any:
    """
        Combine instances from a list of dicts into a single dict, by stacking arrays along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW. It also convert some values to tensors.
        The list of dicts refers to the the list of tiles produced by GridPatch transform applied on a WSI.
    """
    collate_keys = []
    constant_keys = slide_metadata_keys
    for key in batch[0][0].keys():
        if key not in slide_metadata_keys:
            if type(batch[0][0][key]) == np.ndarray:
                collate_keys.append(key)
            else:
                logging.warning(f'Only np.ndarray are collated - {key} value will be taken from first patch')
                constant_keys.append(key)
    tensor_keys = collate_keys + [SlideKey.LABEL]

    skip_idx = check_patch_location_format(batch)
    new_batch: List[dict] = []
    for patch_data in batch:
        # we assume all patches are dictionaries with the same keys
        data = patch_data[0]
        # this is necessary to overcome bug in RandGRidPatch, if one patch has faulty location the all slide is skipped
        if data[SlideKey.SLIDE_ID] not in skip_idx:
            for key in collate_keys:
                if key == SlideKey.PATCH_LOCATION:
                    data[key] = np.array([ix[key] for ix in patch_data if type(ix[key][0]) == np.uint8])
                else:
                    data[key] = np.array([ix[key] for ix in patch_data])
            for key in tensor_keys:
                data[key] = torch.tensor(data[key])
            new_batch.append(data)
            batch = new_batch
    return multibag_collate(batch)
