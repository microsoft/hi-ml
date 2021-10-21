#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Callable

from line_profiler import LineProfiler
from openslide import OpenSlide
from PIL import Image
import cucim
import numpy as np

from azureml.core import Dataset, Run, Workspace

from Histopathology.preprocessing.create_tiles_dataset import (
    process_slide,
    process_slide_cucim_no_save,
    process_slide_open_slide_no_save,
    process_slide_cucim,
    process_slide_openslide,
    save_tile, generate_tiles)


def profile_cucim(input_file: Path,
                  output_file: Path) -> None:
    """
    Load an input_file with cuCIM, print out basic properties, and save as output_file.

    :param input_file: Input file path.
    :param output_file: Output file path.
    :return: None
    """
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
    """
    Load an input_file with OpenSlide, print out basic properties, and save as output_file.

    :param input_file: Input file path.
    :param output_file: Output file path.
    :return: None
    """
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


def profile_folder(mount_point: Path,
                   output_folder: Path,
                   subfolder: str) -> None:
    """
    For each *.tiff image file in the given subfolder or the mount_point,
    load each with cuCIM or OpenSlide, print out basic properties, and save as a png.

    :param mount_point: Base path for source images.
    :param output_folder: Base path to save output images.
    :param subfolder: Subfolder of mount_point to search for tiffs.
    :return: None.
    """
    cucim_output_folder = output_folder / "cc" / subfolder
    cucim_output_folder.mkdir(parents=True, exist_ok=True)

    openslide_output_folder = output_folder / "slide" / subfolder
    openslide_output_folder.mkdir(parents=True, exist_ok=True)

    for image_file in (mount_point / subfolder).glob("*.tiff"):
        output_filename = image_file.with_suffix(".png").name

        try:
            profile_cucim(image_file, cucim_output_folder / output_filename)
        except Exception as ex:
            print(f"Error calling cuCIM: {str(ex)}")
        profile_openslide(image_file, openslide_output_folder / output_filename)


def profile_folders(mount_point: Path,
                    output_folder: Path) -> None:
    profile_folder(mount_point, output_folder, "train_images")
    profile_folder(mount_point, output_folder, "train_label_masks")


def print_cache_state(cache) -> None:  # type: ignore
    """
    Print out cuCIM cache state
    """
    print(f"cache_hit: {cache.hit_count}, cache_miss: {cache.miss_count}")
    print(f"items in cache: {cache.size}/{cache.capacity}, "
          f"memory usage in cache: {cache.memory_size}/{cache.memory_capacity}")


def wrap_profile_folders(mount_point: Path,
                         output_folder: Path) -> None:
    """
    Load some tiffs with cuCIM and OpenSlide, save them, and run line_profile.

    :return: None.
    """
    def wrap_profile_folders():
        profile_folders(mount_point, output_folder)

    lp = LineProfiler()
    lp.add_function(profile_cucim)
    lp.add_function(profile_openslide)
    lp.add_function(profile_folder)
    lp_wrapper = lp(wrap_profile_folders)
    lp_wrapper()
    with open("outputs/profile_folders.txt", "w", encoding="utf-8") as f:
        lp.print_stats(f)


def profile_main(mount_point: Path,
                 output_folder: Path,
                 label: str,
                 process: Callable) -> None:
    def wrap_main():
        from Histopathology.preprocessing.create_tiles_dataset import main
        main(process,
             panda_dir=mount_point,
             root_output_dir=output_folder / label,
             level=1,
             tile_size=224,
             margin=64,
             occupancy_threshold=0.05,
             parallel=False,
             overwrite=True)

    lp = LineProfiler()
    lp.add_function(process_slide)
    lp.add_function(save_tile)
    lp.add_function(generate_tiles)
    lp_wrapper = lp(wrap_main)
    lp_wrapper()
    with open(f"outputs/profile_{label}.txt", "w", encoding="utf-8") as f:
        lp.print_stats(f)


def main() -> None:
    """
    Load some tiffs with cuCIM and OpenSlide, then run image tiling with both libraries.

    :return: None.
    """
    print(f"cucim.is_available(): {cucim.is_available()}")
    print(f"cucim.is_available('skimage'): {cucim.is_available('skimage')}")
    print(f"cucim.is_available('core'): {cucim.is_available('core')}")
    print(f"cucim.is_available('clara'): {cucim.is_available('clara')}")

    cucim.CuImage.cache("per_process", memory_capacity=2048, record_stat=True)
    cache = cucim.CuImage.cache()
    print(f"cucim.cache.config: {cache.config}")
    print_cache_state(cache)

    run_context = Run.get_context()
    if hasattr(run_context, 'experiment'):
        ws = run_context.experiment.workspace
        output_folder = Path("outputs")
    else:
        ws = Workspace.from_config()
        output_folder = Path("../outputs")

    dataset = Dataset.get_by_name(ws, name='panda')

    with dataset.mount("/tmp/datasets/panda") as mount_context:
        mount_point = Path(mount_context.mount_point)

        wrap_profile_folders(mount_point, output_folder)

        print_cache_state(cache)

        for i, process in enumerate([process_slide_cucim_no_save,
                                     process_slide_open_slide_no_save,
                                     process_slide_cucim,
                                     process_slide_openslide]):
            label = f'process_slide_{i}'
            profile_main(mount_point, output_folder, label, process)

            print_cache_state(cache)


if __name__ == '__main__':
    main()
