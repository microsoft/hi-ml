import logging

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Iterable, Type

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, recall_score, roc_auc_score, roc_curve

from health.utils.reports.reports import Plot, Report

BEST_EPOCH_FOLDER_NAME = "best_validation_epoch"
DEFAULT_HUE_KEY = "Default"


@unique
class LoggingColumns(Enum):
    """
    This enum contains string constants that act as column names in logging, and in all files on disk.
    """
    DataSplit = "data_split"
    Patient = "subject"
    Hue = "prediction_target"
    Structure = "structure"
    Dice = "dice"
    HausdorffDistanceMM = "HausdorffDistanceMM"
    Epoch = "epoch"
    Institution = "institutionId"
    Series = "seriesId"
    Tags = "tags"
    AccuracyAtThreshold05 = "accuracy_at_threshold_05"
    Loss = "loss"
    CrossEntropy = "cross_entropy"
    SecondsPerEpoch = "seconds_per_epoch"
    SecondsPerBatch = "seconds_per_batch"
    AreaUnderRocCurve = "area_under_roc_curve"
    AreaUnderPRCurve = "area_under_pr_curve"
    CrossValidationSplitIndex = "cross_validation_split_index"
    ModelOutput = "model_output"
    Label = "label"
    SubjectCount = "subject_count"
    ModelExecutionMode = "model_execution_mode"
    MeanAbsoluteError = "mean_absolute_error"
    MeanSquaredError = "mean_squared_error"
    LearningRate = "learning_rate"
    ExplainedVariance = "explained_variance"
    NumTrainableParameters = "num_trainable_parameters"
    AccuracyAtOptimalThreshold = "accuracy_at_optimal_threshold"
    OptimalThreshold = "optimal_threshold"
    FalsePositiveRateAtOptimalThreshold = "false_positive_rate_at_optimal_threshold"
    FalseNegativeRateAtOptimalThreshold = "false_negative_rate_at_optimal_threshold"
    SequenceLength = "sequence_length"


@dataclass
class Results:
    true_positives: pd.DataFrame
    false_positives: pd.DataFrame
    true_negatives: pd.DataFrame
    false_negatives: pd.DataFrame


@unique
class ModelExecutionMode(Enum):
    """
    Model execution mode
    """
    TRAIN = "Train"
    TEST = "Test"
    VAL = "Val"


def get_optimal_idx(fpr: np.ndarray, tpr: np.ndarray) -> np.ndarray:
    """
        Given a list of FPR and TPR values corresponding to different thresholds, compute the index which corresponds
        to the optimal threshold.
        """
    optimal_idx = np.argmax(tpr - fpr)
    return optimal_idx


def read_csv_and_filter_prediction_target(csv: Path, prediction_target: str,
                                          crossval_split_index: Optional[int] = None,
                                          data_split: Optional[ModelExecutionMode] = None,
                                          epoch: Optional[int] = None,
                                          dtypes=None) -> pd.DataFrame:
    """
    Given one of the CSV files written during inference time, read it and select only those rows which belong to the
    given prediction_target. Also check that the final subject IDs are unique.

    :param csv: Path to the metrics CSV file. Must contain at least the following columns (defined in the LoggingColumns
        enum): LoggingColumns.Patient, LoggingColumns.Hue.
    :param prediction_target: Target ("hue") by which to filter.
    :param crossval_split_index: If specified, filter rows only for the respective run (requires
        LoggingColumns.CrossValidationSplitIndex).
    :param data_split: If specified, filter rows by Train/Val/Test (requires LoggingColumns.DataSplit).
    :param epoch: If specified, filter rows for given epoch (default: last epoch only; requires LoggingColumns.Epoch).
    :return: Filtered dataframe.
    """

    def check_column_present(dataframe: pd.DataFrame, column: LoggingColumns) -> None:
        if column.value not in dataframe:
            raise ValueError(f"Missing {column.value} column.")

    dtypes = dtypes or {'subject': np.int32}
    df = pd.read_csv(csv, dtype=dtypes)
    df['subject'] = df['subject'].astype(int)
    df['epoch'] = df['epoch'].astype(int)
    # df.convert_dtypes
    df = df[df[LoggingColumns.Hue.value] == prediction_target]  # Filter by prediction target
    df = df[~df[LoggingColumns.Label.value].isna()]  # Filter missing labels

    # Filter by crossval split index
    if crossval_split_index is not None:
        check_column_present(df, LoggingColumns.CrossValidationSplitIndex)
        df = df[df[LoggingColumns.CrossValidationSplitIndex.value] == crossval_split_index]

    # Filter by Train/Val/Test
    if data_split is not None:
        check_column_present(df, LoggingColumns.DataSplit)
        df = df[df[LoggingColumns.DataSplit.value] == data_split.value]

    # Filter by epoch
    if LoggingColumns.Epoch.value in df:
        # In a FULL_METRICS_DATAFRAME_FILE, the epoch column will be BEST_EPOCH_FOLDER_NAME (string) for the Test split.
        # Here we cast the whole column to integer, mapping BEST_EPOCH_FOLDER_NAME to -1.
        epochs = df[LoggingColumns.Epoch.value].apply(lambda e: -1 if e == BEST_EPOCH_FOLDER_NAME else int(e))
        if epoch is None:
            epoch = epochs.max()  # Take last epoch if unspecified
        df = df[epochs == epoch]
    elif epoch is not None:
        raise ValueError(f"Specified epoch {epoch} but missing {LoggingColumns.Epoch.value} column.")

    if not df[LoggingColumns.Patient.value].is_unique:
        raise ValueError(f"Subject IDs should be unique, but found duplicate entries "
                         f"in column {LoggingColumns.Patient.value} in the csv file.")
    return df


def get_correct_and_misclassified_examples(val_metrics_csv: Path, test_metrics_csv: Path,
                                           prediction_target: str = DEFAULT_HUE_KEY) -> Optional[Results]:
    """
    Given the paths to the metrics files for the validation and test sets, get a list of true positives,
    false positives, false negatives and true negatives.
    The threshold for classification is obtained by looking at the validation file, and applied to the test set to get
    label predictions.
    The validation and test csvs must have at least the following columns (defined in the LoggingColumns enum):
    LoggingColumns.Hue, LoggingColumns.Patient, LoggingColumns.Label, LoggingColumns.ModelOutput.
    """
    df_val = read_csv_and_filter_prediction_target(val_metrics_csv, prediction_target)

    if len(df_val) == 0:
        return None

    fpr, tpr, thresholds = roc_curve(df_val[LoggingColumns.Label.value], df_val[LoggingColumns.ModelOutput.value])
    optimal_idx = get_optimal_idx(fpr=fpr, tpr=tpr)
    optimal_threshold = thresholds[optimal_idx]

    df_test = read_csv_and_filter_prediction_target(test_metrics_csv, prediction_target)

    if len(df_test) == 0:
        return None

    df_test["predicted"] = df_test.apply(lambda x: int(x[LoggingColumns.ModelOutput.value] >= optimal_threshold),
                                         axis=1)

    true_positives = df_test[(df_test["predicted"] == 1) & (df_test[LoggingColumns.Label.value] == 1)]
    false_positives = df_test[(df_test["predicted"] == 1) & (df_test[LoggingColumns.Label.value] == 0)]
    false_negatives = df_test[(df_test["predicted"] == 0) & (df_test[LoggingColumns.Label.value] == 1)]
    true_negatives = df_test[(df_test["predicted"] == 0) & (df_test[LoggingColumns.Label.value] == 0)]

    return Results(true_positives=true_positives,
                   true_negatives=true_negatives,
                   false_positives=false_positives,
                   false_negatives=false_negatives)


def get_k_best_and_worst_performing(val_metrics_csv: Path, test_metrics_csv: Path, k: int,
                                    prediction_target: str = DEFAULT_HUE_KEY) -> Optional[Results]:
    """
    Get the top "k" best predictions (i.e. correct classifications where the model was the most certain) and the
    top "k" worst predictions (i.e. misclassifications where the model was the most confident).
    """
    results = get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_csv,
                                                     test_metrics_csv=test_metrics_csv,
                                                     prediction_target=prediction_target)
    if results is None:
        return None

    # sort by model_output
    sorted = Results(true_positives=results.true_positives.sort_values(by=LoggingColumns.ModelOutput.value,
                                                                       ascending=False).head(k),
                     true_negatives=results.true_negatives.sort_values(by=LoggingColumns.ModelOutput.value,
                                                                       ascending=True).head(k),
                     false_positives=results.false_positives.sort_values(by=LoggingColumns.ModelOutput.value,
                                                                         ascending=False).head(k),
                     false_negatives=results.false_negatives.sort_values(by=LoggingColumns.ModelOutput.value,
                                                                         ascending=True).head(k))
    return sorted


def add_table_of_best_and_worst_performing(val_metrics_csv: Path,
                                           test_metrics_csv: Path,
                                           k: int,
                                           prediction_target: str,
                                           report: Report):

    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_csv,
                                              test_metrics_csv=test_metrics_csv,
                                              k=k,
                                              prediction_target=prediction_target)

    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_csv,
                                              test_metrics_csv=test_metrics_csv,
                                              k=k,
                                              prediction_target=prediction_target)
    if results is None:
        report.add_header("Empty validation or test set")
        return

    report.add_header(f"Top {len(results.false_positives)} false positives")
    report.add_table_from_dataframe(results.false_positives,
                                    cols_to_print=[LoggingColumns.Patient.value, LoggingColumns.ModelOutput.value],
                                    override_headers=["", "ID", "Score"],
                                    print_index=True)

    report.add_header(f"Top {len(results.false_negatives)} false negatives")
    report.add_table_from_dataframe(results.false_negatives,
                                    cols_to_print=[LoggingColumns.Patient.value, LoggingColumns.ModelOutput.value],
                                    override_headers=["", "ID", "Score"],
                                    print_index=True)

    report.add_header(f"Top {len(results.true_positives)} true positives")
    report.add_table_from_dataframe(results.true_positives,
                                    cols_to_print=[LoggingColumns.Patient.value, LoggingColumns.ModelOutput.value],
                                    override_headers=["", "ID", "Score"],
                                    print_index=True)

    report.add_header(f"Top {len(results.true_negatives)} true negatives")
    report.add_table_from_dataframe(results.true_negatives,
                                    cols_to_print=[LoggingColumns.Patient.value, LoggingColumns.ModelOutput.value],
                                    override_headers=["", "ID", "Score"],
                                    print_index=True)

    report.add_header("Plot best and worst sample images")


