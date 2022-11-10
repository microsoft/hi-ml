#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Utility functions for interacting with mlflow runs
"""
from typing import Any, Dict, List
from mlflow.client import MlflowClient
from mlflow.entities import Run, Metric


def get_mlflow_run(mlflow_client: MlflowClient, mlflow_run_id: str) -> Run:
    """
    Retrieve a Run from an MLFlow client

    :param mlflow_client: An MLflowClient object.
    :param mlflow_run_id: The id of an mlflow run to retrieve.
    :return: An mlflow Run object
    """
    mlflow_run = mlflow_client.get_run(mlflow_run_id)
    return mlflow_run


def get_last_metrics_from_mlflow_run(mlflow_run: Run) -> Dict[str, Any]:
    """
    Retrieve the last logged metrics from an mlflow Run

    :param mlflow_run: the mlflow Run to retrieve metrics from
    :return: A dictionary of metric_name to value
    """
    metrics = mlflow_run.data.metrics
    return metrics


def get_metric_from_mlflow_run(mlflow_client: MlflowClient, run_id: str, metric_name: str
                               ) -> List[Metric]:
    """
    For a given metric name, get the entire history of logged values from an mlflow Run

    :param mlflow_client: An MLFlowClient object.
    :param run_id: The id of the run to retrieve the metrics from.
    :param metric_name: The name of the metric to retrieve values for.
    :return: A list of mlflow Metric objects representing the all of the values of the given
        metric throughout the run
    """
    metric_history = mlflow_client.get_metric_history(run_id, metric_name)
    return metric_history
