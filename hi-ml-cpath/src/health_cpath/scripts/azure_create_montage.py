"""
This script is used to create a montage of slides, given a slides dataset (e.g. TCGA-PRAD, TCGA-BRCA).

To submit to Azure ML, run the following:
`python azure_create_montage.py --azureml`

A json configuration file containing the credentials to the Azure workspace and an environment.yml file are expected
in input.
"""
import logging
from pathlib import Path
import sys
from typing import List, Optional
import pandas as pd
import param


current_file = Path(__file__)
cpath_root = current_file.absolute().parent.parent.parent.parent
sys.path.append(str(cpath_root))

himl_root = cpath_root / "hi-ml"
folders_to_add = [himl_root / "hi-ml" / "src",
                  himl_root / "hi-ml-azure" / "src",
                  himl_root / "hi-ml-cpath" / "src"]
for folder in folders_to_add:
    if folder.is_dir():
        sys.path.insert(0, str(folder))

from health_azure.himl import submit_to_azure_if_needed, DatasetConfig  # noqa
from health_azure.utils import apply_overrides, create_argparser, parse_arguments  # noqa
from health_azure.logging import logging_to_stdout  # noqa
from cpath.datasets.tcga_prad_private_dataset import TcgaPradPrivateDataset  # noqa
from cpath.datasets.tcga_brca_private_dataset import TcgaBrcaPrivateDataset  # noqa
from cpath.utils.montage_utils import (TCGA_BRCA_PREFIX, TCGA_PRAD_PREFIX, DatasetOrDataframe,  # noqa
                                       dataset_from_folder,
                                       montage_from_included_and_excluded_slides)


class MontageConfig(param.Parameterized):
    dataset = \
        param.String(default="",
                     doc="The name of the AzureML dataset to use for creating the montage. The dataset will be "
                         "mounted automatically. Use an absolute path to a folder on the local machine to bypass "
                         "mounting")
    datastore = \
        param.String(default="innereyedatasets",
                     doc="The name of the AzureML datastore where the dataset is defined.")
    conda_env: Optional[Path] = \
        param.ClassSelector(class_=Path, default=Path("hi-ml/hi-ml-cpath/environment.yml"), allow_None=True,
                            doc="The Conda environment file that should be used when submitting the present run to "
                                "AzureML. If not specified, the hi-ml-cpath environment file will be used.")
    level: int = \
        param.Integer(default=1,
                      doc="Resolution downsample level, e.g. if lowest resolution is 40x and the available "
                          "downsample levels are [1.0, 4.0, 16.0] then level = 1 corresponds to 10x magnification")
    cluster: str = \
        param.String(default="", allow_None=False,
                     doc="The name of the GPU or CPU cluster inside the AzureML workspace"
                         "that should execute the job. To run on your local machine, omit this argument.")
    exclude_by_slide_id: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None, allow_None=True,
                            doc="Provide a file that contains slide IDs that should be excluded. File format is "
                                "CSV, the first column is used as the slide ID. If the file is empty, no slides "
                                "will be excluded.")
    include_by_slide_id: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None, allow_None=True,
                            doc="Provide a file that contains slide IDs that should be included. File format is "
                                "CSV, the first column is used as the slide ID. If the file is empty, no montage "
                                "will be produced.")
    image_glob_pattern: str = \
        param.String(default="",
                     doc="When provided, use this pattern in rglob to find the files that should be included in the "
                         "montage. Example: '**/*.tiff' to find all TIFF files recursive. You may have to escape "
                         "the pattern in your shell.")
    width: int = \
        param.Integer(default=60_000,
                      doc="The width of the montage in pixels")
    output_path: Path = \
        param.ClassSelector(class_=Path,
                            default=Path("outputs"),
                            doc="The folder where the montage will be saved")
    parallel: int = \
        param.Integer(default=8,
                      doc="The number of parallel processes to use when creating the montage.")
    backend: str = \
        param.String(default="openslide",
                     doc="The backend to use for reading the slides. Can be 'openslide' or 'cucim'")

    def read_list(self, csv_file_path: Optional[Path]) -> List[str]:
        """Reads a list of slide IDs from a file."""
        if csv_file_path:
            df = pd.read_csv(csv_file_path)
            column_to_read = df.columns[0]
            if len(df.columns) > 1:
                print(f"WARNING: More than one column in file, using first column: {column_to_read}")
            return df[column_to_read].tolist()
        else:
            return []

    def read_exclusion_list(self) -> List[str]:
        """Read the list of slide IDs that should be excluded from the montage."""
        if self.exclude_by_slide_id:
            slides_to_exclude = self.read_list(self.exclude_by_slide_id)
            print(f"Excluding {len(slides_to_exclude)} slides from montage. First 3: {slides_to_exclude[:3]}")
            print("Exclusion list will be matched against the Slide ID column (for predefined datasets) or the "
                  "filename.")
            return slides_to_exclude
        else:
            return []

    def read_inclusion_list(self) -> List[str]:
        """Read the list of slide IDs that should be included in the montage."""
        if self.include_by_slide_id:
            slides_to_include = self.read_list(self.include_by_slide_id)
            print(f"Restricting montage to {len(slides_to_include)} slides. First 3: {slides_to_include[:3]}")
            print("Inclusion list will be matched against the Slide ID column (for predefined datasets) or the "
                  "filename.")
            return slides_to_include
        else:
            return []

    def read_dataset(self, input_folder: Path) -> DatasetOrDataframe:
        """Read the dataset that should be used for creating the montage. If the dataset is recognized as either
        TCGA-PRAD or TCGA-BRCA, the dataset will be read from the predefined dataset. Otherwise, all image files in the
        input folder will be used.

        :param input_folder: The folder where the dataset is located.
        :return: A SlidesDataset or dataframe object that contains the dataset."""
        if self.dataset.startswith(TCGA_PRAD_PREFIX):
            print("The dataset name indicates that its format is TCGA-PRAD, trying to load that.")
            dataset: DatasetOrDataframe = TcgaPradPrivateDataset(input_folder)
        elif self.dataset.startswith(TCGA_BRCA_PREFIX):
            print("The dataset name indicates that its format is TCGA-BRCA, trying to load that.")
            dataset = TcgaBrcaPrivateDataset(input_folder)
        else:
            print("Trying to create a dataset from all files in the input folder.")
            if not self.image_glob_pattern:
                raise ValueError(
                    "When the dataset name does not indicate the dataset type, you must provide a glob "
                    "pattern to find the files that should be included via --image_glob_pattern"
                )
            try:
                dataset = dataset_from_folder(input_folder, glob_pattern=self.image_glob_pattern)
            except Exception as ex:
                raise ValueError(f"Unable to create dataset from files in folder {input_folder}: {ex}")
            if len(dataset) == 0:
                raise ValueError(f"No images found in folder {input_folder} with pattern {self.image_glob_pattern}")
        return dataset


def create_config_from_args() -> MontageConfig:
    config = MontageConfig()
    parser = create_argparser(config,
                              usage="python azure_create_montage.py --dataset <azureml_dataset> "
                                    "--cluster <cluster_name> --level <level> --exclude_by_slide_id <path_to_file> "
                                    "--conda_env <path_to_conda_env_file>",
                              description="Create an overview image with thumbnails of all slides in a dataset.")
    parser_results = parse_arguments(parser, args=sys.argv[1:], fail_on_unknown_args=True)
    _ = apply_overrides(config, parser_results.args)
    return config


def create_montage(config: MontageConfig, input_folder: Path) -> None:
    dataset = config.read_dataset(input_folder)
    config.output_path.mkdir(parents=True, exist_ok=True)
    if config.include_by_slide_id and config.exclude_by_slide_id:
        raise ValueError("You cannot provide both an inclusion and exclusion list.")
    if config.include_by_slide_id:
        items = config.read_inclusion_list()
        exclude_items = False
    elif config.exclude_by_slide_id:
        items = config.read_exclusion_list()
        exclude_items = True
    else:
        items = []
        exclude_items = True
    montage_from_included_and_excluded_slides(dataset=dataset,
                                              items=items,
                                              exclude_items=exclude_items,
                                              output_path=config.output_path,
                                              width=config.width,
                                              num_parallel=config.parallel,
                                              backend=config.backend)


if __name__ == "__main__":
    config = create_config_from_args()
    logging_to_stdout()
    submit_to_azureml = config.cluster != ""
    if config.dataset.strip() == "":
        raise ValueError("Please provide a dataset name via --dataset")
    elif config.dataset.startswith("/"):
        if submit_to_azureml:
            raise ValueError("Cannot submit to AzureML if dataset is a local folder")
        input_folder: Optional[Path] = Path(config.dataset)
    else:
        logging.info(f"In AzureML use mounted dataset '{config.dataset}' in datastore {config.datastore}")
        input_dataset = DatasetConfig(name=config.dataset, datastore=config.datastore, use_mounting=True)
        logging.info(f"Submitting to AzureML, running on cluster {config.cluster}")
        run_info = submit_to_azure_if_needed(entry_script=current_file,
                                             snapshot_root_directory=cpath_root,
                                             compute_cluster_name=config.cluster,
                                             conda_environment_file=config.conda_env,
                                             submit_to_azureml=submit_to_azureml,
                                             input_datasets=[input_dataset],
                                             strictly_aml_v1=True,
                                             docker_shm_size="100g",
                                             )
        input_folder = run_info.input_datasets[0]

    assert input_folder is not None
    create_montage(config, input_folder)
