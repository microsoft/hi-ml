#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import logging
import param
from copy import deepcopy
from health_azure.logging import logging_section
from health_cpath.datasets.base_dataset import SlidesDataset
from health_cpath.preprocessing.tiff_conversion import AMPERSAND, UNDERSCORE, ConvertWSIToTiffd, WSIFormat
from health_cpath.utils.naming import SlideKey
from monai.data.dataset import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, List, Optional
from tifffile.tifffile import COMPRESSION


class TiffConversionConfig(param.Parameterized):
    image_key: SlideKey = param.ClassSelector(
        class_=SlideKey,
        default=SlideKey.IMAGE,
        doc="The key of the image in the dataset. This is used to get the path of the src file."
    )
    src_format: WSIFormat = param.ClassSelector(
        class_=WSIFormat, default=WSIFormat.NDPI, doc="The format of the source files. Default is NDPI."
    )
    target_magnifications: Optional[List[float]] = param.List(
        default=[5.0],
        doc="The magnifications that will be saved in the tiff files. Use None for all available magnifications.",
    )
    add_lowest_magnification: bool = param.Boolean(
        default=False,
        doc="If True, the lowest magnification will be saved in the tiff files in addition to the target "
        "magnifications. If False, only the specified magnifications will be saved. This is especially useful for "
        "costly computations that can be applied at a lower magnification.",
    )
    base_objective_power: Optional[float] = param.Number(
        default=40.0,
        doc="The objective power of the base magnification of the originale whole slide image. This is used to "
        "calculate the levels corresponding to the target magnifications",
    )
    replace_ampersand_by: str = param.String(
        default=UNDERSCORE,
        doc="The character that will replace the ampersand in the file name. & (as in H&E) can be problematic in some "
        "file systems. It is recommended to use _ instead.",
    )
    compression: COMPRESSION = param.ClassSelector(
        default=COMPRESSION.ADOBE_DEFLATE,
        class_=COMPRESSION,
        doc="The compression that will be used for the tiff files. Default is ADOBE_DEFLATE for lossless compression.",
    )
    tile_size: int = param.Integer(
        default=512, doc="The size of the tiles used to write the tiff file, defaults to 512."
    )
    num_workers: int = param.Integer(
        default=1,
        doc="The number of workers that will be used to convert the src files to tiff files. If num_workers is 1.",
    )
    converted_dataset_csv: str = param.String(
        default="",
        doc="The name of the new dataset csv file that will be created for the converted data. If None, the default "
        "name of the original dataset will be used.",
    )

    def get_transform(self, output_folder: Path) -> ConvertWSIToTiffd:
        """Get the transform that will be used to convert the src files to tiff files."""
        return ConvertWSIToTiffd(
            output_folder=output_folder,
            image_key=self.image_key,
            src_format=self.src_format,
            target_magnifications=self.target_magnifications,
            add_lowest_magnification=self.add_lowest_magnification,
            base_objective_power=self.base_objective_power,
            replace_ampersand_by=self.replace_ampersand_by,
            compression=self.compression,
            tile_size=self.tile_size,
        )

    def create_dataset_csv_for_converted_data(self, dataset: SlidesDataset, output_folder: Path) -> None:
        """Create a new dataset csv file for the converted data.

        :param dataset_df: The original dataset csv file.
        :param output_folder: The folder where the new dataset csv file will be saved.
        """
        new_dataset_df = deepcopy(dataset.dataset_df)
        new_dataset_df[dataset.IMAGE_COLUMN] = (
            new_dataset_df[dataset.IMAGE_COLUMN]
            .str.replace(self.src_format, WSIFormat.TIFF)
            .str.replace(AMPERSAND, self.replace_ampersand_by)
        )
        new_dataset_path = output_folder / (self.converted_dataset_csv or dataset.DEFAULT_CSV_FILENAME)
        new_dataset_df.to_csv(new_dataset_path, sep="\t" if new_dataset_path.suffix == ".tsv" else ",")
        logging.info(f"Saved new dataset tsv file to {new_dataset_path}")

    def __call__(self, dataloader: DataLoader) -> None:
        for _ in tqdm(dataloader, total=len(dataloader)):
            pass

    def run(self, dataset: SlidesDataset, output_folder: Path, wsi_subfolder: Optional[str] = None) -> None:
        """Run the conversion of the src files to tiff files.

        dataset: The slides dataset that contains the src wsi.
        output_folder: The folder where the tiff files will be saved.
        image_subfolder: The subfolder where the tiff files will be saved. If None, the tiff files will be saved in the
        root output folder.
        """
        if wsi_subfolder is not None:
            wsi_output_folder = output_folder / wsi_subfolder
            wsi_output_folder.mkdir(parents=True, exist_ok=True)
            logging.info(f"Whole slide images will be saved to subfolder {wsi_output_folder}")
        transformed_dataset = Dataset(dataset, self.get_transform(wsi_output_folder))  # type: ignore
        dataloader = DataLoader(transformed_dataset, num_workers=self.num_workers, batch_size=1)
        with logging_section(f"Starting conversion of {len(dataset)} slides to tiff format to {output_folder}"):
            self(dataloader)
            self.create_dataset_csv_for_converted_data(dataset, output_folder)
