from pathlib import Path
import os

import cucim
import numpy as np
from cucim import CuImage
from openslide import OpenSlide
from pathlib import Path
from PIL import Image

from azureml.core import Dataset, Datastore, Run, Workspace


# @profile
def profile_cucim(input_file: Path,
                  output_file: Path) -> None:
    img = CuImage(str(input_file))

    count = img.resolutions['level_count']
    dimensions = img.resolutions['level_dimensions']

    print(f"level_count: {count}")
    print(f"level_dimensions: {dimensions}")

    print(img.metadata)

    region = img.read_region(location=(0,0),
                             size=dimensions[count-1],
                             level=count-1)
    np_img_arr = np.asarray(region)
    img2 = Image.fromarray(np_img_arr)
    img2.save(output_file)


# @profile
def profile_openslide(input_file: Path,
                      output_file: Path) -> None:
    with OpenSlide(str(input_file)) as img:
        count = img.level_count
        dimensions = img.level_dimensions

        print(f"level_count: {count}")
        print(f"dimensions: {dimensions}")

        for k, v in img.properties.items():
            print(k, v)

        region = img.read_region(location=(0,0),
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


if __name__ == '__main__':
    print(cucim.is_available())
    print(cucim.is_available("skimage"))
    print(cucim.is_available("core"))
    print(cucim.is_available("clara"))

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

        #root_output_dir = output_folder / "tiles"
        #root_output_dir.mkdir(exist_ok=True)

        #from preprocessing.create_tiles_dataset import main
        #main(panda_dir="/tmp/datasets/PANDA",
        #    root_output_dir=root_output_dir,
        #    level=1,
        #    tile_size=224,
        #    margin=64,
        #    occupancy_threshold=0.05,
        #    parallel=False,
        #    overwrite=True)
