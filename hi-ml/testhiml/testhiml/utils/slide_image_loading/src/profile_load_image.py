#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
import os

import cucim
import numpy as np
from line_profiler import LineProfiler
from openslide import OpenSlide
from PIL import Image

from azureml.core import Dataset, Run, Workspace


def profile_cucim(input_file: Path,
                  output_file: Path) -> None:
    img = cucim.CuImage(str(input_file))

    count = img.resolutions['level_count']
    dimensions = img.resolutions['level_dimensions']

    print(f"level_count: {count}")
    print(f"level_dimensions: {dimensions}")

    print(img.metadata)

    region = img.read_region(location=(0, 0),
                             size=dimensions[count-1],
                             level=count-1)
    np_img_arr = np.asarray(region)
    img2 = Image.fromarray(np_img_arr)
    img2.save(output_file)


def profile_openslide(input_file: Path,
                      output_file: Path) -> None:
    with OpenSlide(str(input_file)) as img:
        count = img.level_count
        dimensions = img.level_dimensions

        print(f"level_count: {count}")
        print(f"dimensions: {dimensions}")

        for k, v in img.properties.items():
            print(k, v)

        region = img.read_region(location=(0, 0),
                                 level=count-1,
                                 size=dimensions[count-1])
        region.save(output_file)


def profile_folder(mount_path: Path,
                   output_folder: Path,
                   subfolder: str) -> None:
    cucim_output_folder = output_folder / "cc" / subfolder
    cucim_output_folder.mkdir(parents=True, exist_ok=True)

    openslide_output_folder = output_folder / "slide" / subfolder
    openslide_output_folder.mkdir(parents=True, exist_ok=True)

    for image_file in (mount_path / subfolder).glob("*.tiff"):
        output_filename = image_file.with_suffix(".png").name

        try:
            profile_cucim(image_file, cucim_output_folder / output_filename)
        except Exception as ex:
            print(f"Error calling cucum: {str(ex)}")
        profile_openslide(image_file, openslide_output_folder / output_filename)


def main() -> None:
    print(f"cucim.is_available(): {cucim.is_available()}")
    print(f"cucim.is_available('skimage'): {cucim.is_available('skimage')}")
    print(f"cucim.is_available('core'): {cucim.is_available('core')}")
    print(f"cucim.is_available('clara'): {cucim.is_available('clara')}")

    run_context = Run.get_context()
    if hasattr(run_context, 'experiment'):
        ws = run_context.experiment.workspace
        output_folder = Path("outputs")
    else:
        ws = Workspace.from_config()
        output_folder = Path("../outputs")

    dataset = Dataset.get_by_name(ws, name='panda')

    with dataset.mount("/tmp/datasets/panda") as mount_context:
        mount_point = mount_context.mount_point
        print(f"Mount point: {mount_point}")
        print(os.listdir(mount_point))

        profile_folder(Path(mount_point),
                       output_folder,
                       "train_images")

        profile_folder(Path(mount_point),
                       output_folder,
                       "train_label_masks")

        root_output_dir = output_folder / "tiles"
        root_output_dir.mkdir(exist_ok=True)

        from Histopathology.preprocessing.create_tiles_dataset import (
            main,
            process_slide_cucim_no_save,
            process_slide_open_slide_no_save,
            process_slide_cucim,
            process_slide_openslide)

        for process in [process_slide_cucim_no_save,
                        process_slide_open_slide_no_save,
                        process_slide_cucim,
                        process_slide_openslide]:
            main(process,
                 'process_slide',
                 panda_dir="/tmp/datasets/panda",
                 root_output_dir=root_output_dir,
                 level=1,
                 tile_size=224,
                 margin=64,
                 occupancy_threshold=0.05,
                 parallel=False,
                 overwrite=True)


if __name__ == '__main__':
    from Histopathology.preprocessing.create_tiles_dataset import (
        process_slide_cucim_no_save,
        process_slide_open_slide_no_save,
        process_slide_cucim,
        process_slide_openslide, save_tile, generate_tiles)

    lp = LineProfiler()
    lp.add_function(profile_cucim)
    lp.add_function(profile_openslide)
    lp.add_function(profile_folder)
    lp.add_function(process_slide_cucim_no_save)
    lp.add_function(process_slide_open_slide_no_save)
    lp.add_function(process_slide_cucim)
    lp.add_function(process_slide_openslide)
    lp.add_function(save_tile)
    lp.add_function(generate_tiles)
    lp_wrapper = lp(main)
    lp_wrapper()
    with open("outputs/profile.txt", "w", encoding="utf-8") as f:
        lp.print_stats(f)
