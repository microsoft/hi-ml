
#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
from argparse import ArgumentParser
import logging
import sys
import param
from copy import deepcopy
from health_azure.logging import logging_section
from health_azure.utils import apply_overrides, create_argparser, parse_arguments
from health_cpath.datasets.base_dataset import SlidesDataset
from health_cpath.preprocessing.tiff_conversion import AMPERSAND, UNDERSCORE, ConvertWSIToTiffd, WSIFormat
from health_cpath.utils.montage_config import AzureRunConfig
from health_cpath.utils.naming import SlideKey
from monai.data.dataset import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, List, Optional


class TiffConversionConfig(AzureRunConfig):
    dest_dir: Path = param.ClassSelector(
        class_=Path, default=Path("outputs"), doc="The folder where the new tiff files will be saved."
    )
    src_format: WSIFormat = param.ClassSelector(
        class_=WSIFormat, default=WSIFormat.NDPI, doc="The format of the source files. Default is NDPI."
    )
    image_key: str = param.String(
        default=SlideKey.IMAGE,
        doc="The key of the image in the dataset. This is used to get the path of the src file."
    )
    target_magnifications: Optional[List[float]] = param.List(
        default=[10.],
        doc="The magnifications that will be saved in the tiff files. Use None for all available magnifications."
    )
    add_lowest_magnification: bool = param.Boolean(
        default=False,
        doc="If True, the lowest magnification will be saved in the tiff files additionally. If False, only the "
            "specified magnifications will be saved. This is especially useful for costly computations that can be "
            "applied at a lower magnification."
    )
    replace_ampersand_by: str = param.String(
        default=UNDERSCORE,
        doc="The character that will replace the ampersand in the file name. & (as in H&E) can be problematic in some "
            "file systems. It is recommended to use _ instead."
    )
    num_workers: int = param.Integer(
        default=1,
        doc="The number of workers that will be used to convert the src files to tiff files. If num_workers is 1."
    )

    def __init__(self, dataset: SlidesDataset, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.transform = ConvertWSIToTiffd(
            dest_dir=self.dest_dir,
            image_key=self.image_key,
            src_format=self.src_format,
            target_magnifications=self.target_magnifications,
            add_lowest_magnification=self.add_lowest_magnification,
            replace_ampersand_by=self.replace_ampersand_by,
        )
        self.dataset = dataset

    def run_conversion(self) -> None:
        transformed_dataset = Dataset(self.dataset, self.transform)  # type: ignore
        dataloader = DataLoader(transformed_dataset, num_workers=self.num_workers, batch_size=1)
        with logging_section(f"Starting conversion of {len(self.dataset)} slides to tiff format to {self.dest_dir}"):
            for _ in tqdm(dataloader, total=len(dataloader)):
                pass

    def create_dataset_csv_for_transformed_data(self, dest_file_name: Optional[str]) -> None:
        new_dataset_df = deepcopy(self.dataset.dataset_df)
        new_dataset_df[self.dataset.IMAGE_COLUMN] = new_dataset_df[self.dataset.IMAGE_COLUMN].str.replace(
            self.src_format, WSIFormat.TIFF).str.replace(AMPERSAND, self.replace_ampersand_by)
        new_dataset_path = self.dest_dir / (dest_file_name or self.dataset.DEFAULT_CSV_FILENAME)
        new_dataset_df.to_csv(new_dataset_path, sep="\t" if new_dataset_path.suffix == ".tsv" else ",")
        logging.info(f"Saved new dataset tsv file to {new_dataset_path}")


def create_tiff_conversion_argparser(dataset: SlidesDataset) -> ArgumentParser:
    return create_argparser(
        TiffConversionConfig(dataset),
        usage="python convert_to_tiff.py --dataset <dataset_folder> --image_glob_pattern '**/*.tiff' --width 1000",
        description="Create an overview image with thumbnails of all slides in a dataset.")


def create_tiff_conversion_config_from_args(dataset: SlidesDataset) -> TiffConversionConfig:
    parser = create_tiff_conversion_argparser(dataset)
    config = TiffConversionConfig(dataset)
    parser_results = parse_arguments(parser, args=sys.argv[1:], fail_on_unknown_args=True)
    _ = apply_overrides(config, parser_results.args)
    return config
