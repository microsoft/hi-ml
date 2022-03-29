import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from typing import List


def list_data_collate(batch: List):
    """
        Combine instances from a list of dicts into a single dict, by stacking them along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW
    """

    for i, item in enumerate(batch):
        data = item[0]
        data["image"] = torch.tensor(np.array([ix["image"] for ix in item]))
        batch[i] = data
    return default_collate(batch)
