#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import cv2
from line_profiler import LineProfiler
from PIL import Image
import torch
from torchvision.io.image import read_image

from health_azure import get_workspace
from health_azure.datasets import get_or_create_dataset


def convert_image_opencv(input_filename: Path, output_filename: Path) -> None:
    im = cv2.imread(str(input_filename))
    greyscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_filename), greyscale)


def convert_image_pillow(input_filename: Path, output_filename: Path) -> None:
    """
    Open an image file, convert it to greyscale, and save to another file.

    :param input_filename: Source image file path.
    :param output_filename: Target image file path.
    :return: None.
    """
    im = Image.open(input_filename)
    print(f"Converting file: {input_filename}, format: {im.format}, size: {im.size}, mode: {im.mode}")
    greyscale = im.convert("L")
    print(f"Converted file: {input_filename}, format: {greyscale.format}, size: {greyscale.size}, mode: {greyscale.mode}")
    greyscale.save(output_filename)


def read_image_torch(input_filename: Path) -> torch.Tensor:
    """
    Open an image file, return it as a torch.Tensor of shape [image_width, image_height, 3]

    :param input_filename: Source image file path.
    :return: Image contents as a torch.Tensor.
    """
    return read_image(str(input_filename))


def convert_image_torch(input_filename: Path, output_filename: Path) -> None:
    """
    Open an image file, convert it to greyscale, and save to another file.

    :param input_filename: Source image file path.
    :param output_filename: Target image file path.
    :return: None.
    """
    im = read_image_torch(input_filename)
    print(f"Converting file: {input_filename}, format: {im.format}, size: {im.size}, mode: {im.mode}")
    greyscale = im.convert("L")
    print(f"Converted file: {input_filename}, format: {greyscale.format}, size: {greyscale.size}, mode: {greyscale.mode}")
    greyscale.save(output_filename)


def mount_and_process_folder() -> None:
    """
    Mount a dataset called 'panda_tiles', assumed to contain image files, with file extension png. Load each png file,
    convert to greyscale, and save to a separate folder.

    :return: None.
    """
    ws = get_workspace(aml_workspace=None, workspace_config_path=None)
    dataset = get_or_create_dataset(workspace=ws,
                                    datastore_name='himldatasets',
                                    dataset_name='panda_tiles_rgb')

    with dataset.mount("/tmp/datasets/panda_tiles_rgb") as mount_context:
        input_folder = Path(mount_context.mount_point)

        output_folder = Path("outputs")
        output_folder.mkdir(exist_ok=True)

        output_folder_opencv = output_folder / "opencv"
        output_folder_opencv.mkdir(exist_ok=True)

        output_folder_pillow = output_folder / "pillow"
        output_folder_pillow.mkdir(exist_ok=True)

        output_folder_torch = output_folder / "torch"
        output_folder_torch.mkdir(exist_ok=True)

        for image_file in input_folder.glob("*.png"):
            convert_image_opencv(image_file, output_folder_opencv / image_file.name)
            convert_image_pillow(image_file, output_folder_pillow / image_file.name)
            convert_image_torch(image_file, output_folder_pillow / image_file.name)


def main() -> None:
    """
    Create a LineProfiler and time calls to convert_image, writing results to a text file.
    """
    lp = LineProfiler()
    lp.add_function(convert_image_opencv)
    lp.add_function(convert_image_pillow)
    lp.add_function(read_image_torch)
    lp_wrapper = lp(mount_and_process_folder)
    lp_wrapper()
    with open("outputs/profile.txt", "w", encoding="utf-8") as f:
        lp.print_stats(f)


if __name__ == '__main__':
    main()
