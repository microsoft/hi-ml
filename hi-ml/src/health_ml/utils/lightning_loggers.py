#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import mlflow
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from health_azure import is_running_in_azure_ml
from health_ml.utils.logging import _preprocess_hyperparams
from health_ml.utils.type_annotations import DictStrFloat, DictStrFloatOrFloatList


class StoringLogger(LightningLoggerBase):
    """
    A Pytorch Lightning logger that simply stores the metrics that are written to it.
    Used for diagnostic purposes in unit tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results_per_epoch: Dict[int, DictStrFloatOrFloatList] = {}
        self.hyperparams: Any = None
        # Fields to store diagnostics for unit testing
        self.train_diagnostics: List[Any] = []
        self.val_diagnostics: List[Any] = []
        self.results_without_epoch: List[DictStrFloat] = []

    @rank_zero_only
    def log_metrics(self, metrics: DictStrFloat, step: Optional[int] = None) -> None:
        logging.debug(f"StoringLogger step={step}: {metrics}")
        epoch_name = "epoch"
        if epoch_name not in metrics:
            # Metrics without an "epoch" key are logged during testing, for example
            self.results_without_epoch.append(metrics)
            return
        epoch = int(metrics[epoch_name])
        del metrics[epoch_name]
        for key, value in metrics.items():
            if isinstance(value, int):
                metrics[key] = float(value)
        if epoch in self.results_per_epoch:
            current_results = self.results_per_epoch[epoch]
            for key, value in metrics.items():
                if key in current_results:
                    logging.debug(f"StoringLogger: appending results for metric {key}")
                    current_metrics = current_results[key]
                    if isinstance(current_metrics, list):
                        current_metrics.append(value)
                    else:
                        current_results[key] = [current_metrics, value]
                else:
                    current_results[key] = value
        else:
            self.results_per_epoch[epoch] = metrics  # type: ignore

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        self.hyperparams = params

    def experiment(self) -> Any:
        return None

    def name(self) -> Any:
        return ""

    def version(self) -> int:
        return 0

    @property
    def epochs(self) -> Iterable[int]:
        """
        Gets the epochs for which the present object holds any results.
        """
        return self.results_per_epoch.keys()

    def extract_by_prefix(self, epoch: int, prefix_filter: str = "") -> DictStrFloat:
        """
        Reads the set of metrics for a given epoch, filters them to retain only those that have the given prefix,
        and returns the filtered ones. This is used to break a set
        of results down into those for training data (prefix "Train/") or validation data (prefix "Val/").

        :param epoch: The epoch for which results should be read.
        :param prefix_filter: If empty string, return all metrics. If not empty, return only those metrics that
        have a name starting with `prefix`, and strip off the prefix.
        :return: A metrics dictionary.
        """
        epoch_results = self.results_per_epoch.get(epoch, None)
        if epoch_results is None:
            raise KeyError(f"No results are stored for epoch {epoch}")
        filtered = {}
        for key, value in epoch_results.items():
            assert isinstance(key, str), f"All dictionary keys should be strings, but got: {type(key)}"
            # Add the metric if either there is no prefix filter (prefix does not matter), or if the prefix
            # filter is supplied and really matches the metric name
            if (not prefix_filter) or key.startswith(prefix_filter):
                stripped_key = key[len(prefix_filter):]
                filtered[stripped_key] = value  # type: ignore
        return filtered  # type: ignore

    def to_metrics_dicts(self, prefix_filter: str = "") -> Dict[int, DictStrFloat]:
        """
        Converts the results stored in the present object into a two-level dictionary, mapping from epoch number to
        metric name to metric value. Only metrics where the name starts with the given prefix are retained, and the
        prefix is stripped off in the result.

        :param prefix_filter: If empty string, return all metrics. If not empty, return only those metrics that
        have a name starting with `prefix`, and strip off the prefix.
        :return: A dictionary mapping from epoch number to metric name to metric value.
        """
        return {epoch: self.extract_by_prefix(epoch, prefix_filter) for epoch in self.epochs}


class MLFlowLogger(LightningLoggerBase):
    HYPERPARAMS_NAME = "hyperparams"

    def _start_run(self, mlflow_run_id: Optional[str] = None) -> None:
        """
        Start an MLFlow run. If an existing run id is given,

        :param mlflow_run_id: _description_, defaults to None
        :return: _description_
        """
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_id=mlflow_run_id)
        return run

    def __init__(self, experiment_name: Optional[str] = None, mlflow_uri: Optional[str] = None,
                 run: Optional[str] = None):
        experiment_name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        self._experiment_name = experiment_name
        mlflow_uri = mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
        self.mlflow_uri = mlflow_uri
        run_id = run or os.environ.get("MLFLOW_RUN_ID")
        self.run_id = run_id
        if run_id is not None:
            logging.info(f"Found existing run id: {run_id}")
            self._start_run(mlflow_run_id=run_id)
        logging.info(f"MLFlow logger info: experiment name: {experiment_name}, run id: {run_id} URI: {mlflow_uri}")
        if not is_running_in_azure_ml:
            self._start_run()

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        is_epoch_metric = "epoch" in metrics
        for key, value in metrics.items():
            # Log all epoch-level metrics without the step information
            # All step-level metrics with step
            mlflow.log_metric(key, value, step=None if is_epoch_metric else step)

    @rank_zero_only
    def log_hyperparams(self, params: Union[argparse.Namespace, Dict[str, Any]]) -> None:
        """
        Logs the given model hyperparameters to MLFlow. Namespaces are converted to dictionaries.
        Nested dictionaries are flattened out. The hyperparameters are then written as a table with two columns
        "name" and "value".
        """
        if params is None:
            return
        params_final = _preprocess_hyperparams(params)
        logging.info(f"Attempting to log hyperparameters: {params_final}")
        if self.run_id is not None:
            retrieved_run = mlflow.get_run(run_id=self.run_id)
            run_data = retrieved_run.data
            existing_params = run_data.params
            existing_keys = existing_params.keys()
            new_params = {}

            for key, val in params_final.items():
                if key in existing_params:
                    num_related_keys = len([k for k in existing_keys if key in k])
                    new_key = key + f"_{num_related_keys}"
                    new_params[new_key] = val
                else:
                    new_params[key] = val
            params_final = new_params

        if len(params_final) > 0:
            mlflow.log_params(params_final)

    @property
    def experiment_name(self) -> Optional[str]:
        return self._experiment_name

    @property
    def name(self) -> Any:
        return ""

    @property
    def version(self) -> int:
        return 0

    @rank_zero_only
    def finalize(self, status: str) -> None:
        mlflow.end_run()
