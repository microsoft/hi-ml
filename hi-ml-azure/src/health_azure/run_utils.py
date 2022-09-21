#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
AML functions that to do with Runs/ Environments aren't available in the new SDK
"""
import hashlib
import json
import logging
import param
import re
import shutil
import tempfile
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import pandas as pd

from azure.ai.ml.entities import Environment, Workspace
from azureml.core import Experiment, Run, get_run
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.run import _OfflineRun
from azureml._restclient.constants import RunStatus

from health_azure.utils import (CustomTypeParam, PathOrString, find_file_in_parent_to_pythonpath, get_workspace,
                                is_local_rank_zero, torch_barrier)

EXPERIMENT_RUN_SEPARATOR = ":"
VALID_LOG_FILE_PATHS = [Path("user_logs/std_log.txt"), Path("azureml-logs/70_driver_log.txt")]
RUN_CONTEXT: Run = Run.get_context()
PARENT_RUN_CONTEXT = getattr(RUN_CONTEXT, "parent", None)


class RunIdOrListParam(CustomTypeParam):
    """
    Wrapper class to allow either a List or string inside of a Parameterized object.
    """

    def _validate(self, val: Any) -> None:
        """
        Checks that the input "val" is indeed a non-empty list or string

        :param val: The value to check
        """
        if val is None:
            if not self.allow_None:
                raise ValueError("Value must not be None")
            else:
                return
        if len(val) == 0 or not (isinstance(val, str) or isinstance(val, list)):
            raise ValueError(f"{val} must be an instance of List or string, found {type(val)}")
        super()._validate(val)

    def from_string(self, x: str) -> List[str]:
        """
        Given a string representing one or more run_ids, first attempts to split into a list, and then
        evaluates each item in the list as a genuine run id

        :param x: The string to evaluate
        :return: a list of one or more strings representing run ids
        """
        res = [str(item) for item in x.split(",")]
        return [determine_run_id_type(x) for x in res]


def get_latest_aml_runs_from_experiment(
    experiment_name: str,
    num_runs: int = 1,
    tags: Optional[Dict[str, str]] = None,
    aml_workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
) -> List[Run]:
    """
    Retrieves the experiment <experiment_name> from the identified workspace and returns <num_runs> latest
    runs from it, optionally filtering by tags - e.g. {'tag_name':'tag_value'}

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to create the workspace.

    :param experiment_name: The experiment name to download runs from
    :param num_runs: The number of most recent runs to return
    :param tags: Optional tags to filter experiments by
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional config file containing settings for the AML Workspace
    :return: a list of one or more Azure ML Run objects
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    experiment: Experiment = workspace.experiments[experiment_name]
    return list(islice(experiment.get_runs(tags=tags), num_runs))


def get_run_file_names(run: Run, prefix: str = "") -> List[str]:
    """
    Get the remote path to all files for a given Run which optionally start with a given prefix

    :param run: The AML Run to look up associated files for
    :param prefix: The optional prefix to filter Run files by
    :return: A list of paths within the Run's container
    """
    all_files = run.get_file_names()
    print(f"Selecting files with prefix {prefix}")
    return [f for f in all_files if f.startswith(prefix)] if prefix else all_files


def _download_files_from_run(run: Run, output_dir: Path, prefix: str = "", validate_checksum: bool = False) -> None:
    """
    Download all files for a given AML run, where the filenames may optionally start with a given
    prefix.

    :param run: The AML Run to download associated files for
    :param output_dir: Local directory to which the Run files should be downloaded.
    :param prefix: Optional prefix to filter Run files by
    :param validate_checksum: Whether to validate the content from HTTP response
    """
    run_paths = get_run_file_names(run, prefix=prefix)
    if len(run_paths) == 0:
        prefix_string = f' with prefix "{prefix}"' if prefix else ""
        raise FileNotFoundError(f"No files{prefix_string} were found for run with ID {run.id}")

    for run_path in run_paths:
        output_path = output_dir / run_path
        _download_file_from_run(run, run_path, output_path, validate_checksum=validate_checksum)


def determine_run_id_type(run_or_recovery_id: str) -> str:
    """
    Determine whether a run id is of type "run id" or "run recovery id". Run recovery ideas take the form
    "experiment_name:run_id". If the input
    string takes the format of a run recovery id, only the run id part will be returned. If it is a run id already,
    it will be returned without transformation.

    :param run_or_recovery_id: The id to determine as either a run id or a run recovery id
    :return: A string representing the run id
    """
    if run_or_recovery_id is None:
        raise ValueError("Expected run_id or run_recovery_id but got None")
    parts = run_or_recovery_id.split(EXPERIMENT_RUN_SEPARATOR)
    if len(parts) > 1:
        # return only the run_id, which comes after the colon
        return parts[1]
    return run_or_recovery_id


def get_aml_run_from_run_id(
    run_id: str, aml_workspace: Optional[Workspace] = None, workspace_config_path: Optional[Path] = None
) -> Run:
    """
    Returns an AML Job object, given the run id (run recovery id will also be accepted but is not recommended
    since AML no longer requires the experiment name in order to find the run from a workspace).

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to create the workspace.

    :param run_id: The run id of the run to download. Can optionally be a run recovery id
    :param aml_workspace: Optional AML Workspace object
    :param workspace_config_path: Optional path to config file containing AML Workspace settings
    :return: An Azure ML Job object
    """
    run_id_ = determine_run_id_type(run_id)
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    return workspace.get_run(run_id_)


def download_files_from_run_id(
    run_id: str,
    output_folder: Path,
    prefix: str = "",
    workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
    validate_checksum: bool = False,
) -> None:
    """
    For a given Azure ML run id, first retrieve the Run, and then download all files, which optionally start
    with a given prefix. E.g. if the run creates a folder called "outputs", which you wish to download all
    files from, specify prefix="outputs". To download all files associated with the run, leave prefix empty.

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.

    If function is called in a distributed PyTorch training script, the files will only be downloaded once per node
    (i.e, all process where is_local_rank_zero() == True). All processes will exit this function once all downloads
    are completed.

    :param run_id: The id of the Azure ML Run
    :param output_folder: Local directory to which the run files should be downloaded.
    :param prefix: Optional prefix to filter run files by
    :param workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path to settings for Azure ML Workspace
    :param validate_checksum: Whether to validate the content from HTTP response
    """
    workspace = get_workspace(aml_workspace=workspace, workspace_config_path=workspace_config_path)
    run = get_aml_run_from_run_id(run_id, aml_workspace=workspace)
    _download_files_from_run(run, output_folder, prefix=prefix, validate_checksum=validate_checksum)
    torch_barrier()


def get_driver_log_file_text(run: Run, download_file: bool = True) -> Optional[str]:
    """
    Returns text stored in run log driver file.

    :param run: Run object representing the current run.
    :param download_file: If ``True``, download log file from the run.
    :return: Driver log file text if a file exists, ``None`` otherwise.
    """
    with tempfile.TemporaryDirectory() as tmp_dir_name:

        for log_file_path in VALID_LOG_FILE_PATHS:
            if download_file:
                run.download_files(
                    prefix=str(log_file_path),
                    output_directory=tmp_dir_name,
                    append_prefix=False,
                )
            tmp_log_file_path = tmp_dir_name / log_file_path
            if tmp_log_file_path.is_file():
                return tmp_log_file_path.read_text()

    files_as_str = ', '.join(f"'{log_file_path}'" for log_file_path in VALID_LOG_FILE_PATHS)
    logging.warning(
        "Tried to get driver log file for run {run.id} text when no such file exists. Expected to find "
        f"one of the following: {files_as_str}"
    )
    return None


def _download_file_from_run(
    run: Run, filename: str, output_file: Path, validate_checksum: bool = False
) -> Optional[Path]:
    """
    Download a single file from an Azure ML Run, optionally validating the content to ensure the file is not
    corrupted during download. If running inside a distributed setting, will only attempt to download the file
    onto the node with local_rank==0. This prevents multiple processes on the same node from trying to download
    the same file, which can lead to errors.

    :param run: The AML Run to download associated file for
    :param filename: The name of the file as it exists in Azure storage
    :param output_file: Local path to which the file should be downloaded
    :param validate_checksum: Whether to validate the content from HTTP response
    :return: The path to the downloaded file if local rank is zero, else None
    """
    if not is_local_rank_zero():
        return None

    run.download_file(filename, output_file_path=str(output_file), _validate_checksum=validate_checksum)
    return output_file


def download_file_if_necessary(run: Run, filename: str, output_file: Path, overwrite: bool = False) -> Path:
    """Download any file from an Azure ML run if it doesn't exist locally.

    :param run: AML Run object.
    :param remote_dir: Remote directory from where the file is downloaded.
    :param download_dir: Local directory where to save the downloaded file.
    :param filename: Name of the file to be downloaded (e.g. `"outputs/test_output.csv"`).
    :param overwrite: Whether to force the download even if the file already exists locally.
    :return: Local path to the downloaded file.
    """
    if not overwrite and output_file.exists():
        print("File already exists at", output_file)
    else:
        output_file.parent.mkdir(exist_ok=True, parents=True)
        _download_file_from_run(run, filename, output_file, validate_checksum=True)
        assert output_file.exists()
        print("File is downloaded at", output_file)
    return output_file


def get_tags_from_hyperdrive_run(run: Run, arg_name: str) -> str:
    """
    Given a child Run that was instantiated as part of a HyperDrive run, retrieve the "hyperparameters" tag
    that AML automatically tags it with, and retrieve a specific tag from within that. The available tags are
    determined by the hyperparameters you specified to perform sampling on. E.g. if you defined AML's
    [Grid Sampling](
    https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#grid-sampling)
    over the space {"learning_rate": choice[1, 2, 3]}, each of your 3 child runs will be tagged with
    hyperparameters: {"learning_rate": 0} and so on


    :param run: An AML run object, representing the child of a HyperDrive run
    :param arg_name: The name of the tag that you want to retrieve - representing one of the hyperparameters you
        specified in sampling.
    :return: A string representing the value of the tag, if found.
    """
    return json.loads(run.tags.get("hyperparameters")).get(arg_name)


def aggregate_hyperdrive_metrics(
    child_run_arg_name: str,
    run_id: Optional[str] = None,
    run: Optional[Run] = None,
    keep_metrics: Optional[List[str]] = None,
    aml_workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    For a given HyperDriveRun object, or id of a HyperDriveRun, retrieves the metrics from each of its children and
    then aggregates it. Optionally filters the metrics logged in the Run, by providing a list of metrics to keep.
    Returns a DataFrame where each column is one child run, and each row is a metric logged by that child run.
    For example, for a HyperDrive run with 2 children, where each logs epoch, accuracy and loss, the result
    would look like::

        |              | 0               | 1                  |
        |--------------|-----------------|--------------------|
        | epoch        | [1, 2, 3]       | [1, 2, 3]          |
        | accuracy     | [0.7, 0.8, 0.9] | [0.71, 0.82, 0.91] |
        | loss         | [0.5, 0.4, 0.3] | [0.45, 0.37, 0.29] |

    here each column is one of the splits/ child runs, and each row is one of the metrics you have logged to the run.

    It is possible to log rows and tables in Azure ML by calling run.log_table and run.log_row respectively.
    In this case, the DataFrame will contain a Dictionary entry instead of a list, where the keys are the
    table columns (or keywords provided to log_row), and the values are the table values. E.g.::

        |                | 0                                        | 1                                         |
        |----------------|------------------------------------------|-------------------------------------------|
        | accuracy_table |{'epoch': [1, 2], 'accuracy': [0.7, 0.8]} | {'epoch': [1, 2], 'accuracy': [0.8, 0.9]} |

    It is also possible to log plots in Azure ML by calling run.log_image and passing in a matplotlib plot. In
    this case, the DataFrame will contain a string representing the path to the artifact that is generated by AML
    (the saved plot in the Logs & Outputs pane of your run on the AML portal). E.g.::

        |                | 0                                       | 1                                     |
        |----------------|-----------------------------------------|---------------------------------------|
        | accuracy_plot  | aml://artifactId/ExperimentRun/dcid.... | aml://artifactId/ExperimentRun/dcid...|

    :param child_run_arg_name: the name of the argument given to each child run to denote its position relative
        to other child runs (e.g. this arg could equal 'child_run_index' - then each of your child runs should expect
        to receive the arg '--child_run_index' with a value <= the total number of child runs)
    :param run: An Azure ML HyperDriveRun object to aggregate the metrics from. Either this or run_id must be provided
    :param run_id: The id (type: str) of a parent/ HyperDrive run. Either this or run must be provided.
    :param keep_metrics: An optional list of metric names to filter the returned metrics by
    :param aml_workspace: If run_id is provided, this is an optional AML Workspace object to retrieve the Run from
    :param workspace_config_path: If run_id is provided, this is an optional path to a config containing details of the
        AML Workspace object to retrieve the Run from.
    :return: A Pandas DataFrame containing the aggregated metrics from each child run
    """
    if run is None:
        assert run_id is not None, "Either run or run_id must be provided"
        workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
        run = get_aml_run_from_run_id(run_id, aml_workspace=workspace)
    # assert isinstance(run, HyperDriveRun)
    metrics: DefaultDict = defaultdict()
    for child_run in run.get_children():  # type: ignore
        child_run_metrics = child_run.get_metrics()
        keep_metrics = keep_metrics or child_run_metrics.keys()

        child_run_tag = get_tags_from_hyperdrive_run(child_run, child_run_arg_name)
        for metric_name, metric_val in child_run_metrics.items():
            if metric_name in keep_metrics:
                if metric_name not in metrics:
                    metrics[metric_name] = {}
                metrics[metric_name][child_run_tag] = metric_val

    df = pd.DataFrame.from_dict(metrics, orient="index")
    return df


def get_most_recent_run_id(run_recovery_file: Path) -> str:
    """
    Gets the string name of the most recently executed AzureML run. This is picked up from the `most_recent_run.txt`
    file.

    :param run_recovery_file: The path of the run recovery file
    :return: The run id
    """
    assert run_recovery_file.is_file(), f"No such file: {run_recovery_file}"

    run_id = run_recovery_file.read_text().strip()
    logging.info(f"Read this run ID from file: {run_id}.")
    return run_id


def get_most_recent_run(run_recovery_file: Path, workspace: Workspace) -> Run:
    """
    Gets the name of the most recently executed AzureML run, instantiates a Run object and returns it

    :param run_recovery_file: The path of the run recovery file
    :param workspace: Azure ML Workspace
    :return: The Run
    """
    run_or_recovery_id = get_most_recent_run_id(run_recovery_file)
    return get_aml_run_from_run_id(run_or_recovery_id, aml_workspace=workspace)


class AmlRunScriptConfig(param.Parameterized):
    """
    Base config for a script that handles Azure ML Runs, which can be retrieved with either a run id, latest_run_file,
    or by giving the experiment name (optionally alongside tags and number of runs to retrieve). A config file path can
    also be presented, to specify the Workspace settings. It is assumed that every AML script would have these
    parameters by default. This class can be inherited from if you wish to add additional command line arguments
    to your script (see HimlDownloadConfig and HimlTensorboardConfig for examples)
    """

    latest_run_file: Path = param.ClassSelector(
        class_=Path,
        default=None,
        instantiate=False,
        doc="Optional path to most_recent_run.txt where the ID of the" "latest run is stored",
    )
    experiment: str = param.String(
        default=None, allow_None=True, doc="The name of the AML Experiment that you wish to download Run files from"
    )
    num_runs: int = param.Integer(
        default=1, allow_None=True, doc="The number of runs to download from the " "named experiment"
    )
    config_file: Path = param.ClassSelector(
        class_=Path, default=None, instantiate=False, doc="Path to config.json where Workspace name is defined"
    )
    tags: Dict[str, Any] = param.Dict()
    run: List[str] = RunIdOrListParam(
        default=None,
        allow_None=True,
        doc="Either single or multiple run id(s). Will be stored as a list"
        " of strings. Also supports run_recovery_ids but this is not "
        "recommended",
    )


def _get_runs_from_script_config(script_config: AmlRunScriptConfig, workspace: Workspace) -> List[Run]:
    """
    Given an AMLRunScriptConfig object, retrieve a run id, given the supplied arguments. For example,
    if "run" has been specified, retrieve the AML Run that corresponds to the supplied run id(s). Alternatively,
    if "experiment" has been specified, retrieve "num_runs" (defaults to 1) latest runs from that experiment. If
    neither is supplied, looks for a file named "most_recent_run.txt" in the current directory and its parents.
    If found, reads the latest run id from there are retrieves the corresponding run. Otherwise, raises a ValueError.

    :param script_config: The AMLRunScriptConfig object which contains the parsed arguments
    :param workspace: an AML Workspace object
    :return: a List of one or more retrieved AML Runs
    """
    if script_config.run is None:
        if script_config.experiment is None:
            # default to latest run file
            latest_run_file = find_file_in_parent_to_pythonpath("most_recent_run.txt")
            if latest_run_file is None:
                raise ValueError("Could not find most_recent_run.txt")
            runs = [get_most_recent_run(latest_run_file, workspace)]
        else:
            # get latest runs from experiment
            runs = get_latest_aml_runs_from_experiment(
                script_config.experiment,
                tags=script_config.tags,
                num_runs=script_config.num_runs,
                aml_workspace=workspace,
            )
    else:
        run_ids: List[str]
        run_ids = script_config.run if isinstance(script_config.run, list) else [script_config.run]  # type: ignore
        runs = [get_aml_run_from_run_id(run_id, aml_workspace=workspace) for run_id in run_ids]
    return runs


def download_checkpoints_from_run_id(
    run_id: str,
    checkpoint_path_or_folder: str,
    output_folder: Path,
    aml_workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
) -> None:
    """
    Given an Azure ML run id, download all files from a given checkpoint directory within that run, to
    the path specified by output_path.
    If running in AML, will take the current workspace. Otherwise, if neither aml_workspace nor
    workspace_config_path are provided, will try to locate a config.json file in any of the
    parent folders of the current working directory.

    :param run_id: The id of the run to download checkpoints from
    :param checkpoint_path_or_folder: The path to the either a single checkpoint file, or a directory of
        checkpoints within the run files. If a folder is provided, all files within it will be downloaded.
    :param output_folder: The path to which the checkpoints should be stored
    :param aml_workspace: Optional AML workspace object
    :param workspace_config_path: Optional workspace config file
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    download_files_from_run_id(
        run_id, output_folder, prefix=checkpoint_path_or_folder, workspace=workspace, validate_checksum=True
    )


def is_running_in_azure_ml(aml_run: Run = RUN_CONTEXT) -> bool:
    """
    Returns True if the given run is inside of an AzureML machine, or False if it is on a machine outside AzureML.
    When called without arguments, this functions returns True if the present code is running in AzureML.
    Note that in runs with "compute_target='local'" this function will also return True. Such runs execute outside
    of AzureML, but are able to log all their metrics, etc to an AzureML run.

    :param aml_run: The run to check. If omitted, use the default run in RUN_CONTEXT
    :return: True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    """
    return hasattr(aml_run, "experiment")


def download_files_from_hyperdrive_children(
    run: Run, remote_file_paths: str, local_download_folder: Path, hyperparam_name: str = ""
) -> List[str]:
    """
    Download a specified file or folder from each of the children of an Azure ML Hyperdrive run. For each child
    run, create a separate folder within your report folder, based on the value of whatever hyperparameter
    was being sampled. E.g. if you sampled over batch sizes 10, 50 and 100, you'll see 3 folders in your
    report folder, named 10, 50 and 100 respectively. If remote_file_path represents a path to a folder, the
    entire folder and all the files within it will be downloaded

    :param run: An AML Run object whose type equals "hyperdrive"
    :param remote_file_paths: A string of one or more paths to the content in the Datastore associated with your
        run outputs, separated by commas
    :param local_download_folder: The local folder to download the files to
    :param hyperparam_name: The name of one of the hyperparameters that was sampled during the HyperDrive
        run. This is used to ensure files are downloaded into logically-named folders
    :return: A list of paths to the downloaded files
    """
    if len(hyperparam_name) == 0:
        raise ValueError(
            "To download results from a HyperDrive run you must provide the hyperparameter name" "that was sampled over"
        )

    # For each child run we create a directory in the local_download_folder named after value of the
    # hyperparam sampled for this child.
    downloaded_file_paths = []
    for child_run in run.get_children():
        child_run_index = get_tags_from_hyperdrive_run(child_run, hyperparam_name)
        if child_run_index is None:
            raise ValueError("Child run expected to have the tag {child_run_tag}")

        # The artifact will be downloaded into a child folder within local_download_folder
        # strip any special characters from the hyperparam index name
        local_folder_child_run = local_download_folder / re.sub("[^A-Za-z0-9]+", "", str(child_run_index))
        local_folder_child_run.mkdir(exist_ok=True)
        for remote_file_path in remote_file_paths.split(","):
            download_files_from_run_id(child_run.id, local_folder_child_run, prefix=remote_file_path)
            downloaded_file_path = local_folder_child_run / remote_file_path
            if not downloaded_file_path.exists():
                logging.warning(
                    f"Unable to download the file {remote_file_path} from the datastore associated" "with this run."
                )
            else:
                downloaded_file_paths.append(str(downloaded_file_path))

    return downloaded_file_paths


def replace_directory(source: Path, target: Path) -> None:
    """
    Safely move the contents of a source directory, deleting any files at the target location.

    Because of how Azure ML mounts output folders, it is impossible to move or rename existing files. Therefore, if
    running in Azure ML, this function creates a copy of the contents of `source`, then deletes the original files.

    :param source: Source directory whose contents should be moved.
    :param target: Target directory into which the contents should be moved. If not empty, all of its contents will be
        deleted first.
    """
    if not source.is_dir():
        raise ValueError(f"Source must be a directory, but got {source}")

    if is_running_in_azure_ml():
        if target.exists():
            shutil.rmtree(target)
        assert not target.exists()

        shutil.copytree(source, target)
        shutil.rmtree(source, ignore_errors=True)
    else:
        # Outside of Azure ML, it should be much faster to rename the directory
        # than to copy all contents then delete, especially for large dirs.
        source.replace(target)

    assert target.exists()
    assert not source.exists()


def create_aml_run_object(
    experiment_name: str,
    run_name: Optional[str] = None,
    workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
    snapshot_directory: Optional[PathOrString] = None,
) -> Run:
    """
    Creates an AzureML Run object in the given workspace, or in the workspace given by the AzureML config file.
    This Run object can be used to write metrics to AzureML, upload files, etc, when the code is not running in
    AzureML. After finishing all operations, use `run.flush()` to write metrics to the cloud, and `run.complete()` or
    `run.fail()`.

    Example:
    >>>run = create_aml_run_object(experiment_name="run_on_my_vm", run_name="try1")
    >>>run.log("foo", 1.23)
    >>>run.flush()
    >>>run.complete()

    :param experiment_name: The AzureML experiment that should hold the run that will be created.
    :param run_name: An optional name for the run (this will be used as the display name in the AzureML UI)
    :param workspace: If provided, use this workspace to create the run in. If not provided, use the workspace
        specified by the `config.json` file in the folder or its parent folder(s).
    :param workspace_config_path: If not provided with an AzureML workspace, then load one given the information in this
        config file.
    :param snapshot_directory: The folder that should be included as the code snapshot. By default, no snapshot
        is created (snapshot_directory=None or snapshot_directory=""). Set this to the folder that contains all the
        code your experiment uses. You can use a file .amlignore to skip specific files or folders, akin to .gitignore
    :return: An AzureML Run object.
    """
    actual_workspace = get_workspace(aml_workspace=workspace, workspace_config_path=workspace_config_path)
    exp = Experiment(workspace=actual_workspace, name=experiment_name)
    if snapshot_directory is None or snapshot_directory == "":
        snapshot_directory = tempfile.mkdtemp()
    return exp.start_logging(display_name=run_name, snapshot_directory=str(snapshot_directory))  # type: ignore


def split_recovery_id(id_str: str) -> Tuple[str, str]:
    """
    Splits a run ID into the experiment name and the actual run.
    The argument can be in the format 'experiment_name:run_id',
    or just a run ID like user_branch_abcde12_123. In the latter case, everything before the last
    two alphanumeric parts is assumed to be the experiment name.
    :param id_str: The string run ID.
    :return: experiment name and run name
    """
    components = id_str.strip().split(EXPERIMENT_RUN_SEPARATOR)
    if len(components) > 2:
        raise ValueError(f"recovery_id must be in the format: 'experiment_name:run_id', but got: {id_str}")
    elif len(components) == 2:
        return components[0], components[1]
    else:
        recovery_id_regex = r"^(\w+)_\d+_[0-9a-f]+$|^(\w+)_\d+$"
        match = re.match(recovery_id_regex, id_str)
        if not match:
            raise ValueError(f"The recovery ID was not in the expected format: {id_str}")
        return (match.group(1) or match.group(2)), id_str


def fetch_run(workspace: Workspace, run_recovery_id: str) -> Run:
    """
    Finds an existing run in an experiment, based on a recovery ID that contains the experiment ID and the actual RunId.
    The run can be specified either in the experiment_name:run_id format, or just the run_id.

    :param workspace: the configured AzureML workspace to search for the experiment.
    :param run_recovery_id: The Run to find. Either in the full recovery ID format, experiment_name:run_id or
        just the run_id
    :return: The AzureML run.
    """
    experiment, run = split_recovery_id(run_recovery_id)
    try:
        experiment_to_recover = Experiment(workspace, experiment)
    except Exception as ex:
        raise Exception(f"Unable to retrieve run {run} in experiment {experiment}: {str(ex)}")
    run_to_recover = fetch_run_for_experiment(experiment_to_recover, run)
    logging.info(f"Fetched run #{run_to_recover.number} {run} from experiment {experiment}.")
    return run_to_recover


def fetch_run_for_experiment(experiment_to_recover: Experiment, run_id: str) -> Run:
    """
    Gets an AzureML Run object for a given run ID in an experiment.

    :param experiment_to_recover: an experiment
    :param run_id: a string representing the Run ID of one of the runs of the experiment
    :return: the run matching run_id_or_number; raises an exception if not found
    """
    try:
        return get_run(experiment=experiment_to_recover, run_id=run_id, rehydrate=True)
    except Exception:
        available_runs = experiment_to_recover.get_runs()
        available_ids = ", ".join([run.id for run in available_runs])
        raise Exception(
            f"Run {run_id} not found for experiment: {experiment_to_recover.name}. Available runs are: {available_ids}"
        )


def is_run_and_child_runs_completed(run: Run) -> bool:
    """
    Checks if the given run has successfully completed. If the run has child runs, it also checks if the child runs
    completed successfully.

    :param run: The AzureML run to check.
    :return: True if the run and all child runs completed successfully.
    """

    def is_completed(run_: Run) -> bool:
        status = run_.get_status()
        if run_.status == RunStatus.COMPLETED:
            return True
        logging.info(f"Run {run_.id} in experiment {run_.experiment.name} finished with status {status}.")
        return False

    runs = list(run.get_children())
    runs.append(run)
    return all(is_completed(run) for run in runs)


def get_metrics_for_childless_run(
    run_id: Optional[str] = None,
    run: Optional[Run] = None,
    keep_metrics: Optional[List[str]] = None,
    aml_workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    For a given Run object or id, retrieves the metrics from that Run and returns them as a pandas DataFrame.
    Optionally filters the metrics logged in the Run, by providing a list of metrics to keep. This function
    expects a childless AML Run. If you wish to aggregate metrics for a Run with children (i.e. a HyperDriveRun),
    please use the function ``aggregate_hyperdrive_metrics``.

    :param run: A (childless) Run object to retrieve the metrics from. Either this or run_id must be provided
    :param run_id: The id (type: str) of a (childless) AML Run. Either this or run must be provided.
    :param keep_metrics: An optional list of metric names to filter the returned metrics by
    :param aml_workspace: If run_id is provided, this is an optional AML Workspace object to retrieve the Run from
    :param workspace_config_path: If run_id is provided, this is an optional path to a config containing details of the
        AML Workspace object to retrieve the Run from.
    :return: A Pandas DataFrame containing the metrics
    """
    if run is None:
        assert run_id is not None, "Either run or run_id must be provided"
        workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
        run = get_aml_run_from_run_id(run_id, aml_workspace=workspace)
    if isinstance(run, _OfflineRun):
        logging.warning("Can't get metrics for _OfflineRun object")
        return pd.DataFrame({})
    metrics = {}
    run_metrics = run.get_metrics()  # type: ignore
    keep_metrics = keep_metrics or run_metrics.keys()
    for metric_name, metric_val in run_metrics.items():
        if metric_name in keep_metrics:
            metrics[metric_name] = metric_val
    df = pd.DataFrame.from_dict(metrics, orient="index")
    return df


def _log_conda_dependencies_stats(conda: CondaDependencies, message_prefix: str) -> None:
    """
    Write number of conda and pip packages to logs.

    :param conda: A conda dependencies object
    :param message_prefix: A message to prefix to the log string.
    """
    conda_packages_count = len(list(conda.conda_packages))
    pip_packages_count = len(list(conda.pip_packages))
    logging.info(f"{message_prefix}: {conda_packages_count} conda packages, {pip_packages_count} pip packages")
    logging.debug("  Conda packages:")
    for p in conda.conda_packages:
        logging.debug(f"    {p}")
    logging.debug("  Pip packages:")
    for p in conda.pip_packages:
        logging.debug(f"    {p}")


def create_python_environment(
    conda_environment_file: Path,
    pip_extra_index_url: str = "",
    workspace: Optional[Workspace] = None,
    private_pip_wheel_path: Optional[Path] = None,
    docker_base_image: str = "",
) -> Environment:
    """
    Creates a description for the Python execution environment in AzureML, based on the arguments.
    The environment will have a name that uniquely identifies it (it is based on hashing the contents of the
    Conda file, the docker base image, environment variables and private wheels.

    :param docker_base_image: The Docker base image that should be used when creating a new Docker image.
    :param pip_extra_index_url: If provided, use this PIP package index to find additional packages when building
        the Docker image.
    :param workspace: The AzureML workspace to work in, required if private_pip_wheel_path is supplied.
    :param private_pip_wheel_path: If provided, add this wheel as a private package to the AzureML environment.
    :param conda_environment_file: The file that contains the Conda environment definition.
    """
    conda_dependencies = CondaDependencies(conda_dependencies_file_path=conda_environment_file)
    yaml_contents = conda_environment_file.read_text()
    if pip_extra_index_url:
        # When an extra-index-url is supplied, swap the order in which packages are searched for.
        # This is necessary if we need to consume packages from extra-index that clash with names of packages on
        # pypi
        conda_dependencies.set_pip_option(f"--index-url {pip_extra_index_url}")
        conda_dependencies.set_pip_option("--extra-index-url https://pypi.org/simple")
    # See if this package as a whl exists, and if so, register it with AzureML environment.
    if private_pip_wheel_path is not None:
        if not private_pip_wheel_path.is_file():
            raise FileNotFoundError(f"Cannot add private wheel: {private_pip_wheel_path} is not a file.")
        if workspace is None:
            raise ValueError("To use a private pip wheel, an AzureML workspace must be provided.")
        whl_url = Environment.add_private_pip_wheel(
            workspace=workspace, file_path=str(private_pip_wheel_path), exist_ok=True
        )
        conda_dependencies.add_pip_package(whl_url)
        logging.info(f"Added add_private_pip_wheel {private_pip_wheel_path} to AzureML environment.")
    # Create a name for the environment that will likely uniquely identify it. AzureML does hashing on top of that,
    # and will re-use existing environments even if they don't have the same name.
    # Hashing should include everything that can reasonably change. Rely on hashlib here, because the built-in
    hash_string = "\n".join(
        [
            yaml_contents,
            docker_base_image,
            # Changing the index URL can lead to differences in package version resolution
            pip_extra_index_url,
            # Use the path of the private wheel as a proxy. This could lead to problems if
            # a new environment uses the same private wheel file name, but the wheel has different
            # contents. In hi-ml PR builds, the wheel file name is unique to the build, so it
            # should not occur there.
            str(private_pip_wheel_path),
        ]
    )
    # Python's hash function gives different results for the same string in different python instances,
    # hence need to use hashlib
    sha1 = hashlib.sha1(hash_string.encode("utf8"))
    overall_hash = sha1.hexdigest()[:32]
    unique_env_name = f"HealthML-{overall_hash}"
    env = Environment(name=unique_env_name)
    env.python.conda_dependencies = conda_dependencies
    if docker_base_image:
        env.docker.base_image = docker_base_image
    return env
