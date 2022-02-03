#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pandas as pd

from health_azure.utils import UnitTestWorkspaceWrapper


DEFAULT_WORKSPACE = UnitTestWorkspaceWrapper()


def create_dataset_df() -> pd.DataFrame:
    """
    Create a test dataframe for DATASET_CSV_FILE_NAME.

    :return: Test dataframe.
    """
    dataset_df = pd.DataFrame()
    dataset_df['subject'] = list(range(10))
    dataset_df['seriesId'] = [f"s{i}" for i in range(10)]
    dataset_df['institutionId'] = ["xyz"] * 10
    return dataset_df


def create_metrics_df() -> pd.DataFrame:
    """
    Create a test dataframe for SUBJECT_METRICS_FILE_NAME.

    :return: Test dataframe.
    """
    metrics_df = pd.DataFrame()
    metrics_df['Patient'] = list(range(10))
    metrics_df['Structure'] = ['appendix'] * 10
    metrics_df['Dice'] = [0.5 + i * 0.02 for i in range(10)]
    return metrics_df


def create_comparison_metrics_df() -> pd.DataFrame:
    """
    Create a test dataframe for comparison metrics.

    :return: Test dataframe.
    """
    comparison_metrics_df = pd.DataFrame()
    comparison_metrics_df['Patient'] = list(range(10))
    comparison_metrics_df['Structure'] = ['appendix'] * 10
    comparison_metrics_df['Dice'] = [0.51 + i * 0.02 for i in range(10)]
    return comparison_metrics_df
