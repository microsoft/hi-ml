from copy import deepcopy
import logging
from argparse import ArgumentParser
import math

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from monai.data.dataset import Dataset
from tqdm import tqdm
from cpath.datasets.cyted_schema import CytedSchema
from cpath.datasets.cyted_slides_dataset import CytedSlidesDataset
from health_cpath.preprocessing.loading import WSIBackend
from health_cpath.utils.naming import SlideKey
from monai.data.wsi_reader import WSIReader
from openslide import OpenSlide
from pathlib import Path
from tifffile.tifffile import TiffWriter, PHOTOMETRIC, COMPRESSION
from typing import Any, Dict, List
from monai.transforms import MapTransform


NDPI = "ndpi"
TIFF = "tiff"
CYTED_SRC = "/data/cyted-raw-20221102"
CYTED_DEST = "/data/cyted-he-tiff-10x-20221202"


class CytedDataConverterd(MapTransform):
    def __init__(self, image_key: str, dest_dir: Path, target_magnification: float = 10.) -> None:
        self.image_key = image_key
        self.dest_dir = dest_dir
        self.target_magnification = target_magnification
        self.wsi_reader = WSIReader(WSIBackend.OPENSLIDE)

    def get_target_level(self, wsi_obj: OpenSlide) -> int:
        """Returns the level of the wsi pyramid that is equal to the target resolution."""
        base_objective_power = int(wsi_obj.properties['openslide.objective-power'])
        for level_idx, level_downsample in enumerate(wsi_obj.level_downsamples):
            objective_power = base_objective_power / level_downsample
            if math.isclose(objective_power, self.target_magnification, rel_tol=1e-3):
                return level_idx
        raise ValueError(f"Target resolution {self.target_magnification} not found")

    def get_highest_level(self, wsi_obj: OpenSlide) -> int:
        """Returns the highest resolution level of the wsi pyramid."""
        return len(wsi_obj.level_downsamples) - 1

    def get_level_data(self, wsi_obj: OpenSlide, level: int) -> np.ndarray:
        level_data, _ = self.wsi_reader.get_data(wsi_obj, level=level)
        return level_data.transpose(1, 2, 0)  # convert to HWC as expected by tiffwriter

    def get_levels(self, wsi_obj: Any) -> List[int]:
        target_level = self.get_target_level(wsi_obj)
        highest_level = self.get_highest_level(wsi_obj)
        return [target_level, highest_level]

    def convert_wsi(self, ndpi_path: Path, tiff_path: Path) -> None:
        """Converts a single wsi file from ndpi to tiff format. The tiff file is saved in the tiff_path. If the
        original ndpi file does not have the target resolution, we skip the wsi and return None."""
        wsi_obj = self.wsi_reader.read(ndpi_path)
        try:
            levels = self.get_levels(wsi_obj)
        except ValueError as e:
            logging.warning(f"Skipping {ndpi_path} because {e}")
            return

        resolution_unit = wsi_obj.properties['tiff.ResolutionUnit']
        assert resolution_unit == 'centimeter', f"Resolution unit is not in centimeters: {resolution_unit}"
        um_per_cm = 10000.
        options = dict(
            software='tifffile',
            metadata={'axes': 'YXC'},
            photometric=PHOTOMETRIC.RGB,
            resolutionunit=resolution_unit,
            compression=COMPRESSION.ADOBE_DEFLATE,  # ADOBE_DEFLATE aka ZLIB lossless compression
            tile=(512, 512),
        )
        with TiffWriter(tiff_path, bigtiff=True) as tif:
            for i, level in enumerate(levels):
                # Warning: the level data should be in YXC format, this messes up with compression
                level_data = self.get_level_data(wsi_obj, level)
                um_per_px = self.wsi_reader.get_mpp(wsi_obj, level=level)
                px_per_cm = (um_per_cm / um_per_px[0], um_per_cm / um_per_px[1])
                # the subfiletype parameter is a bitfield that determines if the wsi_level is a reduced version of
                # another image. level 0 (i.e. i=0) is the full resolution image in the pyramid.
                tif.write(level_data, resolution=px_per_cm, subfiletype=int(i > 0), **options)

    def __call__(self, data: Dict) -> Dict:
        ndpi_path = Path(data[self.image_key])
        tiff_path = self.dest_dir / ndpi_path.name.replace(NDPI, TIFF).replace('&', '_')
        if not tiff_path.exists():
            self.convert_wsi(ndpi_path, tiff_path)
        return data


def run_conversion(
    cyted_dataset: CytedSlidesDataset, dest_dir: Path, target_magnification: float, num_workers: int
) -> None:
    """Run the conversion of the cyted dataset from ndpi to tiff."""

    conversion_transform = CytedDataConverterd(SlideKey.IMAGE, dest_dir, target_magnification)
    transformed_dataset = Dataset(cyted_dataset, conversion_transform)  # type: ignore
    dataloader = DataLoader(transformed_dataset, num_workers=num_workers, batch_size=1, collate_fn=None)

    logging.info(f"Starting conversion of {len(cyted_dataset)} slides to tiff format to {dest_dir}")
    for data in tqdm(dataloader, total=len(dataloader)):
        pass
    logging.info("Conversion finished.")


def create_tiff_dataset_csv(dataset: CytedSlidesDataset, dest_dir: Path, dest_file_name: str = "") -> None:
    """Creates a new TSV file with the same schema as the original cyted dataset TSV file, but with the new
    tiff file paths.
    The TSV file is saved in `dest_dir` / `dest_file_name`. If the `dest_file_name` is not given, the default
    TSV file name from the dataset is used."""
    logging.info(f"Creating new TSV file with tiff file paths at {dest_dir}")
    new_tsv: pd.DataFrame = deepcopy(dataset.dataset_df)
    new_tsv[dataset.IMAGE_COLUMN] = new_tsv[dataset.IMAGE_COLUMN].str.replace(NDPI, TIFF).str.replace('&', '_')
    new_tsv_path = dest_dir / (dest_file_name or dataset.DEFAULT_CSV_FILENAME)
    new_tsv.to_csv(new_tsv_path, sep='\t')
    logging.info(f"Saved new dataset tsv file to {new_tsv_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dataset", type=str, default=CYTED_SRC, help="Root directory of the dataset",
    )
    parser.add_argument(
        "--image_column", type=str, default=CytedSchema.HEImage, help="Name of the column containing the image path",
    )
    parser.add_argument(
        "--label_column", type=str, default=CytedSchema.TFF3Positive, help="Name of the column containing the label",
    )
    parser.add_argument(
        "--target_magnification", type=float, default=10.,
        help="Resolution to convert the images to. Must be one of the available resolutions in the dataset.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers to use for the data loader. Default is 1.",
    )
    parser.add_argument(
        "--dest_dir", type=str, default=CYTED_DEST,
        help="Destination directory to save the tiff files.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Number of slides to convert. Default is None.",
    )
    args = parser.parse_args()

    cyted_dataset = CytedSlidesDataset(root=args.root_dataset,
                                       image_column=args.image_column,
                                       label_column=args.label_column)
    if args.limit is not None:
        cyted_dataset.dataset_df = cyted_dataset.dataset_df.iloc[:args.limit]

    run_conversion(cyted_dataset, Path(args.dest_dir), args.target_magnification, args.num_workers)
    create_tiff_dataset_csv(cyted_dataset, Path(args.dest_dir))
