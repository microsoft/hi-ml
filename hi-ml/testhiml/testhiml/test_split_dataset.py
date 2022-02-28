#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from .utils.fixed_paths_for_tests import full_test_data_path
from health_ml.utils.common_utils import ModelExecutionMode
from health_ml.utils.split_dataset import DatasetSplits

CSV_GROUP_HEADER = 'group'
CSV_INSTITUTION_HEADER = 'institution'
CSV_SUBJECT_HEADER = 'subject'
DATASET_CSV_FILE_NAME = 'dataset.csv'


def test_split_by_subject_ids() -> None:
    test_df, test_ids, train_ids, val_ids = _get_test_df()
    splits = DatasetSplits.from_subject_ids(test_df, train_ids, test_ids, val_ids, subject_column=CSV_SUBJECT_HEADER)

    for x, y in zip([splits.train, splits.test, splits.val], [train_ids, test_ids, val_ids]):
        pd.testing.assert_frame_equal(x, test_df[test_df.subject.isin(y)])


@pytest.mark.parametrize("splits", [[[], ['1'], ['2']], [['1'], [], ['2']], [[], [], ['2']]])
def test_split_by_subject_ids_invalid(splits: List[List[str]]) -> None:
    df1 = pd.read_csv(full_test_data_path(suffix=DATASET_CSV_FILE_NAME), dtype=str)
    with pytest.raises(ValueError):
        DatasetSplits.from_subject_ids(df1, train_ids=splits[0], val_ids=splits[1], test_ids=splits[2],
                                       subject_column=CSV_SUBJECT_HEADER)


def test_get_subject_ranges_for_splits() -> None:
    def _check_at_least_one(x: Dict[ModelExecutionMode, Set[str]]) -> None:
        assert all(len(x[mode]) >= 1 for mode in x.keys())

    proportions = [0.5, 0.4, 0.1]

    splits = DatasetSplits.get_subject_ranges_for_splits(['1', '2', '3'],
                                                         proportions[0],
                                                         proportions[1],
                                                         proportions[2])
    _check_at_least_one(splits)

    splits = DatasetSplits.get_subject_ranges_for_splits(['1'], proportions[0], proportions[1], proportions[2])
    assert splits[ModelExecutionMode.TRAIN] == {'1'}

    population = list(map(str, range(100)))
    splits = DatasetSplits.get_subject_ranges_for_splits(population, proportions[0], proportions[1], proportions[2])
    _check_at_least_one(splits)
    assert all(
        [np.isclose(len(splits[mode]) / len(population), proportions[i]) for i, mode in enumerate(splits.keys())])


def _check_is_partition(total: pd.DataFrame, parts: Iterable[pd.DataFrame], column: str) -> None:
    """Asserts that `total` is the union of `parts`, and that the latter are pairwise disjoint"""
    if column is None:
        return
    total = set(total[column].unique())
    parts = [set(part[column].unique()) for part in parts]
    assert total == set.union(*parts)
    for part1, part2 in combinations(parts, 2):
        assert part1.isdisjoint(part2)


@pytest.mark.parametrize("group_column", [None, CSV_GROUP_HEADER, CSV_SUBJECT_HEADER])
def test_grouped_splits(group_column: str) -> None:
    test_df = _get_test_df()[0]
    proportions = [0.5, 0.4, 0.1]
    splits = DatasetSplits.from_proportions(test_df, proportions[0], proportions[1], proportions[2],
                                            group_column=group_column, subject_column=CSV_SUBJECT_HEADER)
    _check_is_partition(test_df, [splits.train, splits.test, splits.val], CSV_SUBJECT_HEADER)
    _check_is_partition(test_df, [splits.train, splits.test, splits.val], group_column)


def _get_test_df() -> Tuple[DataFrame, List[str], List[str], List[str]]:
    test_data = {
        CSV_SUBJECT_HEADER: list(map(str, range(0, 100))),
        CSV_INSTITUTION_HEADER: ([0] * 10) + ([1] * 90),
        CSV_GROUP_HEADER: [i // 5 for i in range(0, 100)],
        "other": list(range(0, 100))
    }
    assert all(np.bincount(test_data[CSV_GROUP_HEADER]) > 1), "Found singleton groups"  # type: ignore
    assert len(np.unique(test_data[CSV_GROUP_HEADER])) > 1, "Found a single group"  # type: ignore
    train_ids, test_ids, val_ids = list(range(0, 50)), list(range(50, 75)), list(range(75, 100))
    test_df = DataFrame(test_data, columns=list(test_data.keys()))
    return test_df, list(map(str, test_ids)), list(map(str, train_ids)), list(map(str, val_ids))
