#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum, unique
from typing import Any, Tuple

import numpy as np
import torch
import torchmetrics as metrics
from torchmetrics.functional import accuracy, auc, auroc, precision_recall_curve, roc
from torchmetrics import Metric


AVERAGE_DICE_SUFFIX = "AverageAcrossStructures"
TRAIN_PREFIX = "train/"
VALIDATION_PREFIX = "val/"


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

    AREA_UNDER_ROC_CURVE = "AreaUnderRocCurve"
    AREA_UNDER_PR_CURVE = "AreaUnderPRCurve"

    # Regression metrics
    MEAN_ABSOLUTE_ERROR = "MeanAbsoluteError"
    MEAN_SQUARED_ERROR = "MeanSquaredError"
    EXPLAINED_VAR = "ExplainedVariance"


class MeanAbsoluteError(metrics.MeanAbsoluteError):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = MetricType.MEAN_ABSOLUTE_ERROR.value

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return self.total > 0  # type: ignore


class MeanSquaredError(metrics.MeanSquaredError):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = MetricType.MEAN_SQUARED_ERROR.value

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return self.total > 0  # type: ignore


class ExplainedVariance(metrics.ExplainedVariance):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = MetricType.EXPLAINED_VAR.value

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return self.n_obs > 0  # type: ignore


class Accuracy05(metrics.Accuracy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = MetricType.ACCURACY_AT_THRESHOLD_05.value

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return (self.total) or (self.tp + self.fp + self.tn + self.fn) > 0  # type: ignore


class ScalarMetricsBase(Metric):
    """
    A base class for all metrics that can only be computed once the complete set of model predictions and labels
    is available. The base class provides an `update` method, and synchronized storage for predictions (field `preds`)
    and labels (field `targets`). Derived classes need to override the `compute` method.
    """

    def __init__(self, name: str = "", compute_from_logits: bool = False):
        super().__init__(dist_sync_on_step=False)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.name = name
        self.compute_from_logits = compute_from_logits

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:  # type: ignore
        self.preds.append(preds)  # type: ignore
        self.targets.append(targets)  # type: ignore

    def compute(self) -> torch.Tensor:
        """
        Computes a metric from the stored predictions and targets.
        """
        raise NotImplementedError("Should be implemented in the child classes")

    def _get_preds_and_targets(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets a tuple of (concatenated predictions, concatenated targets).
        """
        preds, targets = torch.cat(self.preds), torch.cat(self.targets)  # type: ignore

        # Handles the case where we have a binary problem and predictions are specified [1-p, p] as predictions
        # where p is probability of class 1. Instead of just specifying p.
        if preds.dim() == 2 and preds.shape[1] == 2:
            assert preds.shape[0] == targets.shape[0]
            return preds[:, 1], targets

        assert preds.dim() == targets.dim() == 1 and preds.shape[0] == targets.shape[0]
        return preds, targets

    @property
    def has_predictions(self) -> bool:
        """
        Returns True if the present object stores at least 1 prediction (self.update has been called at least once),
        or False if no predictions are stored.
        """
        return len(self.preds) > 0  # type: ignore

    def _get_metrics_at_optimal_cutoff(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the ROC to find the optimal cut-off i.e. the probability threshold for which the
        difference between true positive rate and false positive rate is smallest. Then, computes
        the false positive rate, false negative rate and accuracy at this threshold (i.e. when the
        predicted probability is higher than the threshold the predicted label is 1 otherwise 0).
        :returns: Tuple(optimal_threshold, false positive rate, false negative rate, accuracy)
        """
        preds, targets = self._get_preds_and_targets()
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan)
        fpr, tpr, thresholds = roc(preds, targets)
        assert isinstance(fpr, torch.Tensor)
        assert isinstance(tpr, torch.Tensor)
        assert isinstance(thresholds, torch.Tensor)
        optimal_idx = torch.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        acc = accuracy(preds > optimal_threshold, targets)
        false_negative_optimal = 1 - tpr[optimal_idx]
        false_positive_optimal = fpr[optimal_idx]
        return optimal_threshold, false_positive_optimal, false_negative_optimal, acc


class AreaUnderRocCurve(ScalarMetricsBase):
    """
    Computes the area under the receiver operating curve (ROC).
    """

    def __init__(self) -> None:
        super().__init__(name=MetricType.AREA_UNDER_ROC_CURVE.value)

    def compute(self) -> torch.Tensor:
        preds, targets = self._get_preds_and_targets()
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan)
        return auroc(preds, targets)


class AreaUnderPrecisionRecallCurve(ScalarMetricsBase):
    """
    Computes the area under the precision-recall-curve.
    """

    def __init__(self) -> None:
        super().__init__(name=MetricType.AREA_UNDER_PR_CURVE.value)

    def compute(self) -> torch.Tensor:
        preds, targets = self._get_preds_and_targets()
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan)
        prec, recall, _ = precision_recall_curve(preds, targets)
        return auc(recall, prec)  # type: ignore
