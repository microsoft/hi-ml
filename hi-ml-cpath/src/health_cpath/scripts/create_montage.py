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
from typing import Optional


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
from health_azure.logging import logging_to_stdout  # noqa
from health_cpath.utils.montage_utils import create_config_from_args  # noqa


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
    config.create_montage(input_folder)
