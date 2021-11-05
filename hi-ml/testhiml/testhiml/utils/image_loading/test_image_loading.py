#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import cv2
from line_profiler import LineProfiler
from PIL import Image

from azureml.core import Dataset

from health_azure import get_workspace


def convert_image_opencv(input_filename: Path, output_filename: Path) -> None:
    im = cv2.imread(input_filename)
    greyscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_filename, greyscale)


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
    greyscale.save(output_filename)


def mount_and_process_folder() -> None:
    """
    Mount a dataset called 'panda_tiles', assumed to contain image files, with file extension png. Load each png file,
    convert to greyscale, and save to a separate folder.

    :return: None.
    """
    ws = get_workspace(aml_workspace=None, workspace_config_path=None)
    dataset = Dataset.get_by_name(ws, name='panda_tiles')

    with dataset.mount("/tmp/datasets/panda_tiles") as mount_context:
        input_folder = Path(mount_context.mount_point)

        output_folder = Path("outputs")
        output_folder.mkdir(exist_ok=True)

        for image_file in input_folder.glob("*.png"):
            output_image_file = output_folder / image_file.name
            convert_image_opencv(image_file)
            convert_image_pillow(image_file, output_image_file)


def main() -> None:
    """
    Create a LineProfiler and time calls to convert_image, writing results to a text file.
    """
    lp = LineProfiler()
    lp.add_function(convert_image_opencv)
    lp.add_function(convert_image_pillow)
    lp_wrapper = lp(mount_and_process_folder)
    lp_wrapper()
    with open("outputs/profile.txt", "w", encoding="utf-8") as f:
        lp.print_stats(f)


if __name__ == '__main__':
    main()
