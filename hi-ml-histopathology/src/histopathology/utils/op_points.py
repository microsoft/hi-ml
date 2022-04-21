from dataclasses import dataclass
from typing import Union

import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve


@dataclass(frozen=True)
class ConfusionMatrix:
    num_total: int
    num_positives: int

    true_positives: np.ndarray
    false_positives: np.ndarray
    thresholds: np.ndarray

    def __post_init__(self) -> None:
        if self.num_total < 0:
            raise ValueError(f"num_total must be > 0, got {self.num_total}")
        if self.num_positives < 0:
            raise ValueError(f"num_positives must be > 0, got {self.num_positives}")
        if self.num_positives > self.num_total:
            raise ValueError(f"num_positives must be <= num_total, got {self.num_positives} > {self.num_total}")

        num_thresholds = len(self.thresholds)
        expected_shape = (num_thresholds,)
        if self.thresholds.shape != expected_shape:
            raise ValueError(f"Expected thresholds with shape {expected_shape}, got {self.thresholds.shape}")
        if self.true_positives.shape != expected_shape:
            raise ValueError(f"Expected true_positives with shape {expected_shape}, got {self.true_positives.shape}")
        if self.false_positives.shape != expected_shape:
            raise ValueError(f"Expected false_positives with shape {expected_shape}, got {self.false_positives.shape}")

        if np.any(self.true_positives > self.num_positives):
            raise ValueError("true_positives must be <= num_positives")
        if np.any(self.false_positives > self.num_negatives):
            raise ValueError("false_positives must be <= num_negatives")

        if not _is_sorted(self.thresholds, ascending=True):
            raise ValueError("thresholds must be in ascending order")
        if not _is_sorted(self.true_positives, ascending=True):
            raise ValueError("true_positives must be in ascending order")
        if not _is_sorted(self.false_positives, ascending=False):
            raise ValueError("false_positives must be in descending order")

    @staticmethod
    def from_labels_and_scores(true_labels: np.ndarray, pred_scores: np.ndarray) -> 'ConfusionMatrix':
        num_total: int = true_labels.shape[0]
        num_positives: int = true_labels.sum()

        fps, tps, thr = _binary_clf_curve(true_labels, pred_scores)
        # Append endpoints to complete the curves
        true_positives: np.ndarray = np.r_[0, tps]
        false_positives: np.ndarray = np.r_[0, fps]
        thresholds: np.ndarray = np.r_[thr[0], thr]
        return ConfusionMatrix(num_total=num_total,
                               num_positives=num_positives,
                               true_positives=true_positives,
                               false_positives=false_positives,
                               thresholds=thresholds)

    @property
    def num_negatives(self) -> int:
        return self.num_total - self.num_positives

    @property
    def false_negatives(self) -> np.ndarray:
        return self.num_positives - self.true_positives

    @property
    def true_negatives(self) -> np.ndarray:
        return self.num_negatives - self.false_positives

    @property
    def pred_positives(self) -> np.ndarray:
        return self.true_positives + self.false_positives

    @property
    def pred_negatives(self) -> np.ndarray:
        return self.num_total - self.pred_positives

    @property
    def prevalence(self) -> float:
        return self.num_positives / self.num_total

    @property
    def sensitivity(self) -> np.ndarray:
        return self.pred_positives / self.num_positives

    @property
    def specificity(self) -> np.ndarray:
        return self.pred_negatives / self.num_negatives

    @property
    def pos_pred_value(self) -> np.ndarray:
        return self.true_positives / self.pred_positives

    @property
    def neg_pred_value(self) -> np.ndarray:
        return self.true_negatives / self.pred_negatives

    def __getitem__(self, index_or_slice: Union[int, slice]) -> 'ConfusionMatrix':
        return ConfusionMatrix(num_total=self.num_total,
                               num_positives=self.num_positives,
                               true_positives=self.true_positives[index_or_slice],
                               false_positives=self.false_positives[index_or_slice],
                               thresholds=self.thresholds[index_or_slice])

    def at_threshold(self, threshold: float) -> 'ConfusionMatrix':
        thr_index = _searchsorted(self.thresholds, threshold, ascending=True)
        return self[thr_index]

    def at_sensitivity(self, sens_level: float) -> 'ConfusionMatrix':
        sens_index = _searchsorted(self.sensitivity, sens_level, ascending=True)
        return self[sens_index]

    def at_specificity(self, spec_level: float) -> 'ConfusionMatrix':
        spec_index = _searchsorted(self.specificity, spec_level, ascending=False)
        return self[spec_index]


def _searchsorted(array: np.ndarray, value: float, ascending: bool) -> int:
    if ascending:
        return np.searchsorted(array, value, side='left')  # type: ignore
    else:
        return len(array) - np.searchsorted(array[::-1], value, side='right')  # type: ignore


def _is_sorted(array: np.ndarray, ascending: bool) -> bool:
    if ascending:
        return np.all(array[:-1] <= array[1:])  # type: ignore
    else:
        return np.all(array[:-1] >= array[1:])  # type: ignore
