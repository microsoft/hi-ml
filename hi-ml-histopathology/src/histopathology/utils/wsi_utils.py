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
    faulty_slides_idx = []
    for i, slide_data in enumerate(batch):
        # TODO check dimension of faulty slide id here
        for patch in slide_data:
            #if patch[SlideKey.SLIDE_ID] == '8d5860e10e09ee25e066ee7fb699453d':
            #    print(patch[WSIPatchKeys.LOCATION])
            #    print(len(patch[WSIPatchKeys.LOCATION]))
            location = patch[WSIPatchKeys.LOCATION]
            if len(location) < 3:
                print(f'Slide {patch[SlideKey.SLIDE_ID]} '
                      f'will be skipped as its patches contained unexpected values in patch_location {location}')
                faulty_slides_idx.append(patch[SlideKey.SLIDE_ID])
                break
    n = len(faulty_slides_idx)
    if n > 0:
        print(f'{n} slides will be skipped because somethign was wrong in the patch location')
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
    for i, patch_data in enumerate(batch):
        data = patch_data[0]
        if data[SlideKey.SLIDE_ID] not in skip_idx:
            for key in collate_keys:
                # if not forcing a type, dtpe will be inferred as np.object in cases where the input image is
                # anomalous (eg. nan values). This will raise an error when converting to tensor.
                data[key] = np.array([ix[key] for ix in patch_data])
            for key in tensor_keys:
                data[key] = torch.tensor(data[key])
            batch[i] = data
    return multibag_collate(batch)
