#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import matplotlib.image as mpimg
import numpy as np
from line_profiler import LineProfiler
from PIL import Image
import torch
from torch.functional import Tensor
import torchvision.transforms.functional as TF
from torchvision.io.image import read_image

from health_azure import get_workspace
from health_azure.datasets import get_or_create_dataset


def crop_size(width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Given an image size, return a test box, as a tuple: (left, top, right, bottom).

    :param width: Image width.
    :param height: Image height.
    :return: Test image crop box.
    """
    left = width / 10
    top = 0
    right = 9 * width / 10
    bottom = height
    return (round(left), top, round(right), bottom)


def convert_pillow(im: Image.Image, greyscale: bool, crop: bool) -> Image.Image:
    """
    Using Pillow optionally crop a vertical section out of an image, then optionally convert it to greyscale.

    :param im: Source image.
    :param greyscale: Optionally convert to greyscale.
    :param crop: Optionally crop.
    :return: Optionally cropped, optionally greyscale image.
    """
    if crop:
        width, height = im.size
        box = crop_size(width, height)
        im = im.crop(box)
    if greyscale:
        im = im.convert("L")
    return im


def read_image_matplotlib(input_filename: Path, greyscale: bool, crop: bool) -> torch.Tensor:
    """
    Read an image file with matplotlib and return a torch.Tensor.

    :param input_filename: Source image file path.
    :param greyscale: Optionally convert to greyscale.
    :param crop: Optionally crop.
    :return: torch.Tensor of shape (C, H, W).
    """
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
    # im is a numpy.array of shape: (H, W), (H, W, 3), or (H, W, 4)
    # where H = height, W = width
    im = mpimg.imread(input_filename)
    if len(im.shape) == 2:
        # if loaded a greyscale image, then it is of shape (H, W) so add in an extra axis
        im = np.expand_dims(im, 2)
    if crop:
        height = im.shape[0]
        width = im.shape[1]
        box = crop_size(width, height)
        im = im[box[1]: box[3], box[0]: box[2], :]
    if greyscale and im.shape[2] >= 3:
        im = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])
        im = np.float32(im) / 255.0
        im = np.expand_dims(im, 2)
    # transpose to shape (C, H, W)
    im = np.transpose(im, (2, 0, 1))
    im_tensor = torch.from_numpy(im)
    return im_tensor


def read_image_matplotlib2(input_filename: Path, greyscale: bool, crop: bool) -> torch.Tensor:
    """
    Read an image file with matplotlib and return a torch.Tensor.

    :param input_filename: Source image file path.
    :param greyscale: Optionally convert to greyscale.
    :param crop: Optionally crop.
    :return: torch.Tensor of shape (C, H, W).
    """
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
    # im is a numpy.array of shape: (H, W), (H, W, 3), or (H, W, 4)
    # where H = height, W = width
    im = mpimg.imread(input_filename)
    im_tensor = TF.to_tensor(im)
    if crop:
        height = im.shape[1]
        width = im.shape[2]
        box = crop_size(width, height)
        im_tensor = TF.crop(im_tensor, box[1], box[0], box[3] - box[1], box[2] - box[0])
    if greyscale and im.shape[0] >= 3:
        im_tensor = TF.rgb_to_grayscale(im_tensor)
    return im_tensor


def read_image_opencv(input_filename: Path, greyscale: bool, crop: bool) -> torch.Tensor:
    """
    Read an image file with OpenCV and return a torch.Tensor.

    :param input_filename: Source image file path.
    :param greyscale: Optionally convert to greyscale.
    :param crop: Optionally crop.
    :return: torch.Tensor of shape (C, H, W).
    """
    # https://docs.opencv.org/4.5.3/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    # im is a numpy.ndarray
    im = cv2.imread(str(input_filename))
    is_greyscale = False not in ((im[:, :, 0] == im[:, :, 1]) == (im[:, :, 1] == im[:, :, 2]))
    if is_greyscale:
        im = im[:, :, 0]
    if len(im.shape) == 2:
        # if loaded a greyscale image, then it is of shape (H, W) so add in an extra axis
        im = np.expand_dims(im, 2)
    if crop:
        height = im.shape[0]
        width = im.shape[1]
        box = crop_size(width, height)
        im = im[box[1]: box[3], box[0]: box[2], :]
    if greyscale and im.shape[2] >= 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = np.expand_dims(im, 2)
    im = np.float32(im) / 255.0
    # transpose to shape (C, H, W)
    im = np.transpose(im, (2, 0, 1))
    im_tensor = torch.from_numpy(im)
    return im_tensor


def read_image_pillow(input_filename: Path, greyscale: bool, crop: bool) -> torch.Tensor:
    """
    Read an image file with pillow and return a torch.Tensor.

    :param input_filename: Source image file path.
    :param greyscale: Optionally convert to greyscale.
    :param crop: Optionally crop.
    :return: torch.Tensor of shape (C, H, W).
    """
    im = Image.open(input_filename)
    if crop:
        width, height = im.size
        box = crop_size(width, height)
        im = im.crop(box)
    if greyscale:
        im = im.convert("L")
    im_tensor = TF.to_tensor(im)
    return im_tensor


def read_image_torch(input_filename: Path, greyscale: bool, crop: bool) -> Optional[torch.Tensor]:
    """
    Read an image file with Torch and return a torch.Tensor.

    :param input_filename: Source image file path.
    :param greyscale: Optionally convert to greyscale.
    :param crop: Optionally crop.
    :return: torch.Tensor of shape (C, H, W).
    """
    try:
        im = read_image(str(input_filename))
    except Exception as e:
        print(f"Problem loading torchvision.io.image.read_image: {e}")
        return None
    if greyscale:
        im = im.convert("L")
    return im


def mount_and_process_folder() -> None:
    """
    Mount a dataset called 'panda_tiles', assumed to contain image files, with file extension png. Load each png file,
    convert to greyscale, and save to a separate folder.

    :return: None.
    """
    source_options: List[Tuple[str, bool, bool]] = [
        ("load", False, False),
        ("greyscale", True, False),
        ("crop", False, True),
        ("crop_greyscale", True, True),
    ]

    target_options: List[Tuple[str, bool, bool]] = [
        ("load", False, False),
    ]

    libs: List[Tuple[str, Callable[[Path, bool, bool], Optional[Tensor]]]] = [
        ("matplotlib", read_image_matplotlib),
        ("matplotlib2", read_image_matplotlib2),
        ("opencv", read_image_opencv),
        ("pillow", read_image_pillow),
        # ("torch", read_image_torch),
    ]

    workspace = get_workspace(aml_workspace=None, workspace_config_path=None)

    dataset = get_or_create_dataset(workspace=workspace,
                                    datastore_name='himldatasets',
                                    dataset_name='panda_tiles')

    with dataset.mount("/tmp/datasets/panda_tiles") as mount_context:
        input_folder = Path(mount_context.mount_point)

        output_folder = Path("outputs")
        output_folder.mkdir(exist_ok=True)

        for option, greyscale, crop in source_options:
            output_folder_name = output_folder / option
            output_folder_name.mkdir(exist_ok=True)

            for image_file in input_folder.glob("*.png"):
                im = Image.open(image_file)
                print(f"Converting file: {image_file}, format: {im.format}, size: {im.size}, mode: {im.mode}")
                im = convert_pillow(im, greyscale, crop)
                print(f"Converted file: {image_file}, format: {im.format}, size: {im.size}, mode: {im.mode}")
                im.save(output_folder_name / image_file.name)

    for repeats in range(0, 10):
        for lib, op in libs:
            output_folder_lib = output_folder / lib
            output_folder_lib.mkdir(exist_ok=True)

            for option, greyscale, crop in target_options:
                output_folder_lib_option = output_folder_lib / option
                output_folder_lib_option.mkdir(exist_ok=True)

                for source_option, _, _ in source_options:
                    source_folder = output_folder / source_option

                    for image_file in source_folder.glob("*.png"):
                        im = Image.open(image_file)
                        source_greyscale = im.mode == 'L'
                        width, height = im.size
                        print(f"Testing file: {image_file}, format: {im.format}, size: {im.size}, mode: {im.mode}")
                        im = op(image_file, greyscale, crop)
                        assert isinstance(im, Tensor)
                        assert im.dtype == torch.float32
                        assert len(im.shape) == 3
                        if crop:
                            box = crop_size(width, height)
                            assert im.shape[2] == box[2] - box[0]
                            assert im.shape[1] == box[3] - box[1]
                        else:
                            assert im.shape[2] == width
                            assert im.shape[1] == height
                        if greyscale or source_greyscale:
                            assert im.shape[0] == 1
                        else:
                            assert im.shape[0] == 3
                        assert torch.max(im) <= 1.0
                        assert torch.min(im) >= 0.0


def main() -> None:
    """
    Create a LineProfiler and time calls to convert_image, writing results to a text file.
    """
    lp = LineProfiler()
    lp.add_function(read_image_matplotlib)
    lp.add_function(read_image_matplotlib2)
    lp.add_function(read_image_opencv)
    lp.add_function(read_image_pillow)
    lp.add_function(read_image_torch)
    lp_wrapper = lp(mount_and_process_folder)
    lp_wrapper()
    with open("outputs/profile.txt", "w", encoding="utf-8") as f:
        lp.print_stats(f)


if __name__ == '__main__':
    main()
