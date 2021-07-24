#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.

See examples/elevate_this.py for a very simple 'hello world' example of use.
"""

import logging
import os
import re
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional

from azureml.core import (Environment, Experiment, Run, RunConfiguration,
                          ScriptRunConfig, Workspace)
from health.azure.datasets import (AzureRunInformation, StrOrDatasetConfig,
                                   _input_dataset_key, _output_dataset_key,
                                   _replace_string_datasets)
from src.health.azure.himl_configs import (SourceConfig, WorkspaceConfig,
                                           get_authentication)

logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)


WORKSPACE_CONFIG_JSON = "config.json"
AZUREML_COMMANDLINE_FLAG = "--azureml"
RUN_CONTEXT = Run.get_context()


def is_running_in_azure(aml_run: Run = RUN_CONTEXT) -> bool:
    """
    Returns True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    :param aml_run: The run to check. If omitted, use the default run in RUN_CONTEXT
    :return: True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    """
    return hasattr(aml_run, 'experiment')


def submit_to_azure_if_needed(
        entry_script: Path,  # type: ignore
        compute_cluster_name: str,
        conda_environment_file: Path,
        workspace_config: Optional[WorkspaceConfig] = None,
        workspace_config_path: Optional[Path] = None,
        snapshot_root_directory: Optional[Path] = None,
        script_params: Optional[List[str]] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        ignored_folders: Optional[List[Path]] = None,
        default_datastore: str = "",
        input_datasets: Optional[List[StrOrDatasetConfig]] = None,
        output_datasets: Optional[List[StrOrDatasetConfig]] = None,
        num_nodes: int = 1,
        ) -> AzureRunInformation:
    """
    Submit a folder to Azure, if needed and run it.

    Use the flag --azureml to submit to AzureML, and leave it out to run locally.

    :param workspace_config: Optional workspace config.
    :param workspace_config_file: Optional path to workspace config file.
    :return: Run object for the submitted AzureML run.
    """
    cleaned_input_datasets = _replace_string_datasets(input_datasets or [],
                                                      default_datastore_name=default_datastore)
    cleaned_output_datasets = _replace_string_datasets(output_datasets or [],
                                                       default_datastore_name=default_datastore)
    in_azure = is_running_in_azure()
    if in_azure:
        returned_input_datasets = [RUN_CONTEXT.input_datasets[_input_dataset_key(index)]
                                   for index in range(len(cleaned_input_datasets))]
        returned_output_datasets = [RUN_CONTEXT.output_datasets[_output_dataset_key(index)]
                                    for index in range(len(cleaned_output_datasets))]
        return AzureRunInformation(
            input_datasets=returned_input_datasets,
            output_datasets=returned_output_datasets,
            run=RUN_CONTEXT,
            is_running_in_azure=True,
            output_folder=Path.cwd() / "outputs",
            log_folder=Path.cwd() / "logs"
        )
    if AZUREML_COMMANDLINE_FLAG not in sys.argv[1:]:
        return AzureRunInformation(
            input_datasets=[d.local_folder for d in cleaned_input_datasets],
            output_datasets=[d.local_folder for d in cleaned_output_datasets],
            run=RUN_CONTEXT,
            is_running_in_azure=False,
            output_folder=Path.cwd() / "outputs",
            log_folder=Path.cwd() / "logs"
        )
    if workspace_config_path and workspace_config_path.is_file():
        auth = get_authentication()
        workspace = Workspace.from_config(path=workspace_config_path, auth=auth)
    elif workspace_config:
        workspace = workspace_config.get_workspace()
    else:
        raise ValueError("Cannot glean workspace config from parameters, and so not submitting to AzureML")
    logging.info(f"Loaded: {workspace.name}")

    environment = Environment.from_conda_specification("simple-env", conda_environment_file)

    # TODO: InnerEye.azure.azure_runner.submit_to_azureml does work here with interupt handlers to kill interupted jobs.
    # We'll do that later if still required.

    run_config = RunConfiguration(
        script=entry_script,
        arguments=script_params)
    run_config.environment = environment

    script_run_config = ScriptRunConfig(
        source_directory=str(snapshot_root_directory),
        run_config=run_config,
        compute_target=workspace.compute_targets[compute_cluster_name],
        environment=environment)
    source_config = SourceConfig(
        snapshot_root_directory=snapshot_root_directory,
        conda_environment_file=conda_environment_file,
        entry_script=entry_script,
        script_params=[p for p in sys.argv[1:] if p != AZUREML_COMMANDLINE_FLAG],
        environment_variables=environment_variables)

    with append_to_amlignore(
            dirs_to_append=ignored_folders or [],
            snapshot_root_directory=snapshot_root_directory or Path.cwd()):
        # TODO: InnerEye.azure.azure_runner.submit_to_azureml does work here with interupt handlers to kill interupted
        # jobs. We'll do that later if still required.

        entry_script_relative_path = \
            source_config.entry_script.relative_to(source_config.snapshot_root_directory).as_posix()
        run_config = RunConfiguration(
            script=entry_script_relative_path,
            arguments=source_config.script_params)
        inputs = {}
        for index, d in enumerate(cleaned_input_datasets):
            consumption = d.to_input_dataset(workspace=workspace, dataset_index=index)
            inputs[consumption.name] = consumption
        outputs = {}
        for index, d in enumerate(cleaned_output_datasets):
            out = d.to_output_dataset(workspace=workspace, dataset_index=index)
            outputs[out.name] = out
        run_config.data = inputs
        run_config.output_data = outputs
        script_run_config = ScriptRunConfig(
            source_directory=str(source_config.snapshot_root_directory),
            run_config=run_config,
            compute_target=workspace.compute_targets[compute_cluster_name])

        # replacing everything apart from a-zA-Z0-9_ with _, and replace multiple _ with a single _.
        experiment_name = re.sub('_+', '_', re.sub(r'\W+', '_', entry_script.stem))
        experiment = Experiment(workspace=workspace, name=experiment_name)

        run: Run = experiment.submit(script_run_config)

        if script_params:
            run.set_tags({"commandline_args": " ".join(script_params)})

        logging.info("\n==============================================================================")
        logging.info(f"Successfully queued new run {run.id} in experiment: {experiment.name}")
        logging.info("Experiment URL: {}".format(experiment.get_portal_url()))
        logging.info("Run URL: {}".format(run.get_portal_url()))
        logging.info("==============================================================================\n")
        exit(0)


@contextmanager
def append_to_amlignore(amlignore: Path, lines_to_append: List[str]) -> Generator:
    """
    Context manager that appends lines to the .amlignore file, and reverts to the previous contents after.
    """
    old_contents = amlignore.read_text() if amlignore.exists() else ""
    new_contents = old_contents.splitlines() + lines_to_append
    amlignore.write_text("\n".join(new_contents))
    yield
    if old_contents:
        amlignore.write_text(old_contents)
    else:
        amlignore.unlink()


def package_setup_and_hacks() -> None:
    """
    Set up the Python packages where needed. In particular, reduce the logging level for some of the used
    libraries, which are particularly talkative in DEBUG mode. Usually when running in DEBUG mode, we want
    diagnostics about the model building itself, but not for the underlying libraries.
    It also adds workarounds for known issues in some packages.
    """
    # Urllib3 prints out connection information for each call to write metrics, etc
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('msrest').setLevel(logging.INFO)
    # AzureML prints too many details about logging metrics
    logging.getLogger('azureml').setLevel(logging.INFO)
    # This is working around a spurious error message thrown by MKL, see
    # https://github.com/pytorch/pytorch/issues/37377
    os.environ['MKL_THREADING_LAYER'] = 'GNU'


def main() -> None:
    """
    Handle submit_to_azure if called from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("-w", "--workspace_name", type=str, required=False, help="Azure ML workspace name")
    parser.add_argument("-s", "--subscription_id", type=str, required=False, help="AzureML subscription id")
    parser.add_argument("-r", "--resource_group", type=str, required=False, help="AzureML resource group")
    parser.add_argument("-p", "--workspace_config_path", type=str, required=False, help="AzureML workspace config file")
    parser.add_argument("-c", "--compute_cluster_name", type=str, required=True, help="AzureML cluster name")
    parser.add_argument("-y", "--snapshot_root_directory", type=str, required=True,
                        help="Root of snapshot to upload to AzureML")
    parser.add_argument("-t", "--entry_script", type=str, required=True,
                        help="The script to run in AzureML")
    parser.add_argument("-d", "--conda_environment_file", type=str, required=True, help="The environment to use")

    args = parser.parse_args()
    if args.workspace_config_path:
        config = None
    else:
        config = WorkspaceConfig(
            workspace_name=args.workspace_name,
            subscription_id=args.subscription_id,
            resource_group=args.resource_group)

    submit_to_azure_if_needed(
        workspace_config=config,
        workspace_config_path=args.workspace_config_path,
        compute_cluster_name=args.compute_cluster_name,
        snapshot_root_directory=args.snapshot_root_directory,
        entry_script=args.entry_script,
        conda_environment_file=args.conda_environment_file)


if __name__ == "__main__":
    main()
