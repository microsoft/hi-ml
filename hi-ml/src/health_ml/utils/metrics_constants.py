#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum, unique
from typing import Union

TRAIN_PREFIX = "train/"
VALIDATION_PREFIX = "val/"


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


@unique
class MetricType(Enum):
    """
    Contains the different metrics that are computed.
    """
    # Any result of loss computation, depending on what's configured in the model.
    LOSS = "Loss"

    # Classification metrics
    CROSS_ENTROPY = "CrossEntropy"
    # Classification accuracy assuming that posterior > 0.5 means predicted class 1
    ACCURACY_AT_THRESHOLD_05 = "AccuracyAtThreshold05"
    ACCURACY_AT_OPTIMAL_THRESHOLD = "AccuracyAtOptimalThreshold"
    # Metrics for segmentation
    DICE = "Dice"
    HAUSDORFF_mm = "HausdorffDistance_millimeters"
    MEAN_SURFACE_DIST_mm = "MeanSurfaceDistance_millimeters"
    VOXEL_COUNT = "VoxelCount"
    PROPORTION_FOREGROUND_VOXELS = "ProportionForegroundVoxels"

    PATCH_CENTER = "PatchCenter"

    AREA_UNDER_ROC_CURVE = "AreaUnderRocCurve"
    AREA_UNDER_PR_CURVE = "AreaUnderPRCurve"
    OPTIMAL_THRESHOLD = "OptimalThreshold"
    FALSE_POSITIVE_RATE_AT_OPTIMAL_THRESHOLD = "FalsePositiveRateAtOptimalThreshold"
    FALSE_NEGATIVE_RATE_AT_OPTIMAL_THRESHOLD = "FalseNegativeRateAtOptimalThreshold"

    # Regression metrics
    MEAN_ABSOLUTE_ERROR = "MeanAbsoluteError"
    MEAN_SQUARED_ERROR = "MeanSquaredError"
    EXPLAINED_VAR = "ExplainedVariance"

    # Common metrics
    SUBJECT_COUNT = "SubjectCount"
    LEARNING_RATE = "LearningRate"


MetricTypeOrStr = Union[str, MetricType]


class TrackedMetrics(Enum):
    """
    Known metrics that are tracked as part of Hyperdrive runs.
    """
    Val_Loss = VALIDATION_PREFIX + MetricType.LOSS.value
