#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import logging
import argparse
import medmnist
import numpy as np

import torch.utils.data as data
import torchvision.transforms as transforms

from torch import Tensor
from medmnist import INFO
from scipy import ndimage
from tifffile import imwrite
from typing import Iterable, List
from health_azure.utils import is_local_rank_zero
from health_ml.utils.common_utils import logging_to_stdout


def save_mock_wsi_as_tiff_file(file_name: str, series: List[np.ndarray]) -> None:
    for i, serie in enumerate(series):
        imwrite(file_name, serie, photometric="rgb", bigtiff=True, compression="zlib", append=(i > 0))


def create_stitched_patches(
    patches: Tensor, sample_counter: int, img_size: int, n_channels: int, step_size: int
) -> np.ndarray:
    mock_image = np.full(shape=(n_channels, img_size, img_size), fill_value=1, dtype=np.float32)
    for i, patch in enumerate(patches):
        mock_image[:, step_size * i: step_size * (i + 1), step_size * i: step_size * (i + 1)] = np.tile(patch, (2, 2))
        np.save(os.path.join(str(sample_counter), f"patch_{i}.npy"), patch.numpy())
    return np.transpose(mock_image, (1, 2, 0))


def create_multi_resolution_wsi(mock_image: np.ndarray, n_series: int, zoom_factor: float) -> List[np.ndarray]:
    series = [ndimage.zoom(mock_image, (1 + i * zoom_factor, 1 + i * zoom_factor, 1)) for i in range(n_series)]
    return series


def create_mock_wsis(
    patch_size: int = 28,
    n_patches: int = 2,
    n_repeat: int = 4,
    n_channels: int = 3,
    n_samples: int = 4,
    n_series: int = 3,
    zoom_factor: float = 0.1,
) -> None:

    data_loader = get_pathmnist_data_loader(n_repeat)
    for sample_counter in range(n_samples):
        os.makedirs(str(sample_counter), exist_ok=True)
        patches, _ = next(iter(data_loader))
        mock_image = create_stitched_patches(
            patches,
            sample_counter,
            img_size=n_patches * n_repeat * patch_size,
            n_channels=n_channels,
            step_size=n_patches * patch_size,
        )
        series = create_multi_resolution_wsi(mock_image, n_series, zoom_factor)
        save_mock_wsi_as_tiff_file(os.path.join(str(sample_counter), "wsi.tiff"), series)


def get_pathmnist_data_loader(batch_size: int = 4) -> Iterable[Tensor]:
    info = INFO["pathmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = DataClass(split="train", transform=data_transform, download=True)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_to_stdout("INFO" if is_local_rank_zero() else "ERROR")
    parser.add_argument("--patch-size", type=int, default=28)
    parser.add_argument("--n-patches", type=int, default=2)
    parser.add_argument("--n-repeat", type=int, default=4)
    parser.add_argument("--n-channels", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-series", type=int, default=3)
    parser.add_argument("--zoom-factor", type=float, default=0.1)
    args = parser.parse_args()
    logging.info(f"Creating {args.n_samples} mock WSIs")
    create_mock_wsis(
        args.patch_size, args.n_patches, args.n_repeat, args.n_channels, args.n_samples, args.n_series, args.zoom_factor
    )
