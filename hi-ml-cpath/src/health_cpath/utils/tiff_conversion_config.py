#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import logging
import param
from copy import deepcopy
from health_azure.logging import logging_section
from health_cpath.datasets.base_dataset import SlidesDataset
from health_cpath.preprocessing.tiff_conversion import AMPERSAND, TIFF_EXTENSION, UNDERSCORE, ConvertWSIToTiffd
from health_cpath.utils.naming import SlideKey
from monai.data.dataset import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional
from tifffile.tifffile import COMPRESSION


class TiffConversionConfig(param.Parameterized):
    image_key: SlideKey = param.ClassSelector(
        class_=SlideKey,
        default=SlideKey.IMAGE,
        doc="The key of the image in the dataset. This is used to get the path of the src file.",
    )
    target_magnifications: Optional[List[float]] = param.List(
        default=[5.0],
        class_=float,
        doc="The magnifications that will be saved in the tiff files. Use None for all available magnifications.",
    )
    add_lowest_magnification: bool = param.Boolean(
        default=False,
        doc="If True, the lowest magnification will be saved in the tiff files in addition to the target "
        "magnifications. If False, only the specified magnifications will be saved. This is especially useful for "
        "costly computations that can be applied at a lower magnification.",
    )
    default_base_objective_power: Optional[float] = param.Number(
        default=None,
        doc="The objective power of the base magnification of the originale whole slide image. This is used to "
        "calculate the levels corresponding to the target magnifications. If None, the objective power will be "
        "extracted from the properties of the src file.",
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
        doc="The number of workers that will be used to convert the src files to tiff files. Defaults to 1.",
    )
    converted_dataset_csv: str = param.String(
        default="",
        doc="The name of the new dataset csv file that will be created for the converted data. If None, the default "
        "name of the original dataset will be used.",
    )
    min_file_size: int = param.Integer(
        default=0,
        doc="The minimum size of the tiff file in bytes. If the tiff file is smaller than this size, it will get "
        "overwritten. Defaults to 0.",
    )
    verbose: bool = param.Boolean(
        default=False,
        doc="If True, the progress of the conversion will be logged including src and tiff file sizes. "
        "Defaults to False.",
    )

    def get_transform(self, output_folder: Path) -> ConvertWSIToTiffd:
        """Get the transform that will be used to convert the src files to tiff files."""
        return ConvertWSIToTiffd(
            output_folder=output_folder,
            image_key=self.image_key,
            target_magnifications=self.target_magnifications,
            add_lowest_magnification=self.add_lowest_magnification,
            default_base_objective_power=self.default_base_objective_power,
            replace_ampersand_by=self.replace_ampersand_by,
            compression=self.compression,
            tile_size=self.tile_size,
            min_file_size=self.min_file_size,
            verbose=self.verbose,
        )

    def create_dataset_csv_for_converted_data(self, output_folder: Path) -> None:
        """Create a new dataset csv file for the converted data.

        :param dataset_df: The original dataset csv file.
        :param output_folder: The folder where the new dataset csv file will be saved.
        """
        new_dataset_df = deepcopy(self.slides_dataset.dataset_df)
        new_dataset_df[self.slides_dataset.image_column] = (
            new_dataset_df[self.slides_dataset.image_column]
            .str.replace(AMPERSAND, self.replace_ampersand_by)
            .map(lambda x: str(Path(x).with_suffix(TIFF_EXTENSION)))
        )
        new_dataset_file = (
            self.converted_dataset_csv if self.converted_dataset_csv else self.slides_dataset.default_csv_filename
        )
        new_dataset_path = output_folder / new_dataset_file
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
        self.slides_dataset: SlidesDataset = dataset
        if wsi_subfolder is not None:
            wsi_output_folder = output_folder / wsi_subfolder
            wsi_output_folder.mkdir(parents=True, exist_ok=True)
            logging.info(f"Whole slide images will be saved to subfolder {wsi_output_folder}")
        else:
            wsi_output_folder = output_folder
        transformed_dataset = Dataset(dataset, self.get_transform(wsi_output_folder))  # type: ignore
        dataloader = DataLoader(transformed_dataset, num_workers=self.num_workers, batch_size=1)
        with logging_section(f"Conversion of {len(dataset)} slides to tiff format to {output_folder}"):
            self(dataloader)
            self.create_dataset_csv_for_converted_data(output_folder)
