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
from tifffile import TiffWriter
from typing import Iterable, List
from health_azure.utils import is_local_rank_zero
from health_ml.utils.common_utils import logging_to_stdout


def save_mock_wsi_as_tiff_file(file_name: str, series: List[np.ndarray]) -> None:
    with TiffWriter(file_name, bigtiff=True) as tif:
        options = dict(photometric="rgb", compression="zlib")
        for i, serie in enumerate(series):
            tif.write(serie, **options, subfiletype=int(i > 0))


def create_pathmnist_stitched_tiles(
    tiles: Tensor, sample_counter: int, img_size: int, n_channels: int, step_size: int, different_tiles: bool = False
) -> np.ndarray:
    mock_image = np.full(shape=(n_channels, img_size, img_size), fill_value=255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        tile = tiles[0] if not different_tiles else tile
        _tile = (tile.numpy() * 255).astype(np.uint8)
        mock_image[:, step_size * i : step_size * (i + 1), step_size * i : step_size * (i + 1)] = np.tile(_tile, (2, 2))
        if different_tiles:
            np.save(os.path.join("pathmnist", f"_{sample_counter}", f"tile_{i}.npy"), _tile)
        elif i == 0:
            np.save(os.path.join("pathmnist", f"_{sample_counter}_tile.npy"), _tile)
    return np.transpose(mock_image, (1, 2, 0))


def create_multi_resolution_wsi(mock_image: np.ndarray, n_series: int) -> List[np.ndarray]:
    series = [mock_image[:: 2 ** i, :: 2 ** i] for i in range(n_series)]
    return series


def get_pathmnist_data_loader(batch_size: int = 4) -> Iterable[Tensor]:
    info = INFO["pathmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset = DataClass(split="train", transform=data_transform, download=True)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def create_pathmnist_mock_wsis(
    tile_size: int = 28,
    n_tiles: int = 2,
    n_repeat: int = 4,
    n_channels: int = 3,
    n_samples: int = 4,
    n_series: int = 3,
    different_tiles: bool = False,
) -> None:

    data_loader = get_pathmnist_data_loader(n_repeat)
    for sample_counter in range(n_samples):
        if different_tiles:
            os.makedirs(f"pathmnist/_{sample_counter}", exist_ok=True)
        tiles, _ = next(iter(data_loader))
        mock_image = create_pathmnist_stitched_tiles(
            tiles,
            sample_counter,
            img_size=n_tiles * n_repeat * tile_size,
            n_channels=n_channels,
            step_size=n_tiles * tile_size,
            different_tiles=different_tiles,
        )
        series = create_multi_resolution_wsi(mock_image, n_series)
        save_mock_wsi_as_tiff_file(os.path.join("pathmnist", f"_{sample_counter}.tiff"), series)


def create_fake_stitched_tiles(
    img_size: int, n_channels: int, step_size: int, n_repeat: int, fill_val: int
) -> np.ndarray:
    mock_image = np.full(shape=(n_channels, img_size, img_size), fill_value=1, dtype=np.uint8)
    for i in range(n_repeat):
        mock_image[:, step_size * i : step_size * (i + 1), step_size * i : step_size * (i + 1)] = fill_val * (i + 1)
    return np.transpose(mock_image, (1, 2, 0))


def create_fake_mock_wsis(
    tile_size: int = 28,
    n_tiles: int = 2,
    n_repeat: int = 4,
    n_channels: int = 3,
    n_samples: int = 4,
    n_series: int = 3,
) -> None:

    for sample_counter in range(n_samples):
        mock_image = create_fake_stitched_tiles(
            img_size=n_tiles * n_repeat * tile_size,
            n_channels=n_channels,
            step_size=n_tiles * tile_size,
            n_repeat=n_repeat,
            fill_val=np.random.randint(0, 60),
        )
        series = create_multi_resolution_wsi(mock_image, n_series)
        save_mock_wsi_as_tiff_file(os.path.join("fake", f"_{sample_counter}.tiff"), series)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_to_stdout("INFO" if is_local_rank_zero() else "ERROR")
    parser.add_argument("--tile-size", type=int, default=28)
    parser.add_argument("--n-tiles", type=int, default=2)
    parser.add_argument("--n-repeat", type=int, default=4)
    parser.add_argument("--n-channels", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-series", type=int, default=3)
    parser.add_argument("--different-tiles", type=bool, default=False)
    parser.add_argument(
        "--mock_type",
        type=str,
        default="pathmnist",
        help="Mock data type: pathmnist for tiles from pathmnist dataset, fake for tiles with fake values",
    )
    args = parser.parse_args()
    logging.info(f"Creating {args.n_samples} mock WSIs")
    if args.mock_type == "pathmnist":
        create_pathmnist_mock_wsis(
            args.tile_size,
            args.n_tiles,
            args.n_repeat,
            args.n_channels,
            args.n_samples,
            args.n_series,
            args.different_tiles,
        )
    elif args.mock_type == "fake":
        create_fake_mock_wsis(
            args.tile_size, args.n_tiles, args.n_repeat, args.n_channels, args.n_samples, args.n_series
        )
