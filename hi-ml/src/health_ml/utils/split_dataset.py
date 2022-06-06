#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from math import ceil
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from health_ml.utils import common_utils
from health_ml.utils.common_utils import ModelExecutionMode


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    subject_column: Optional[str] = None
    group_column: Optional[str] = None
    allow_empty: bool = False

    def __post_init__(self) -> None:
        common_utils.check_properties_are_not_none(self)

        def pairwise_intersection(*collections: Iterable) -> Set:
            """
            Returns any element that appears in more than one collection

            :return: a Set of elements that appear in more than one collection
            """
            intersection = set()
            for col1, col2 in combinations(map(set, collections), 2):
                intersection |= col1 & col2
            return intersection

        # perform dataset split validity assertions
        unique_train, unique_test, unique_val = self.unique_subjects()
        intersection = pairwise_intersection(unique_train, unique_test, unique_val)

        if len(intersection) != 0:
            raise ValueError("Train, Test, and Val splits must have no intersection, found: {}".format(intersection))

        if self.group_column is not None:
            groups_train = self.train[self.group_column].unique()
            groups_test = self.test[self.group_column].unique()
            groups_val = self.val[self.group_column].unique()
            group_intersection = pairwise_intersection(groups_train, groups_test, groups_val)
            if len(group_intersection) != 0:
                raise ValueError("Train, Test, and Val splits must have no intersecting groups, found: {}"
                                 .format(group_intersection))

        if (not self.allow_empty) and any([len(x) == 0 for x in [unique_train, unique_val]]):
            raise ValueError("train_ids({}), val_ids({}) must have at least one value"
                             .format(len(unique_train), len(unique_val)))

    def __str__(self) -> str:
        unique_train, unique_test, unique_val = self.unique_subjects()
        return f'Train: {len(unique_train)}, Test: {len(unique_test)}, and Val: {len(unique_val)}. ' \
               f'Total subjects: {len(unique_train) + len(unique_test) + len(unique_val)}'

    def unique_subjects(self) -> Tuple[Any, Any, Any]:
        """
        Return a tuple of pandas Series of unique subjects across train, test and validation data splits,
        based on self.subject_column

        :return: a tuple of pandas Series
        """
        return (self.train[self.subject_column].unique(),
                self.test[self.subject_column].unique(),
                self.val[self.subject_column].unique())

    def number_of_subjects(self) -> int:
        """
        Returns the sum of unique subjects in the dataset (identified by self.subject_column), summed
        over train, test and validation data splits

        :return: An integer representing the number of unique subjects
        """
        unique_train, unique_test, unique_val = self.unique_subjects()
        return len(unique_train) + len(unique_test) + len(unique_val)

    def __getitem__(self, mode: ModelExecutionMode) -> pd.DataFrame:
        """
        Retrieve either the train, validation or test data in the form of a Pandas dataframe, depending
        on the current execution mode

        :param mode: The current ModelExecutionMode
        :return: A dataframe of the relevant data split
        """
        if mode == ModelExecutionMode.TRAIN:
            return self.train
        elif mode == ModelExecutionMode.TEST:
            return self.test
        elif mode == ModelExecutionMode.VAL:
            return self.val
        else:
            raise ValueError(f"Model execution mode not recognized: {mode}")

    @staticmethod
    def get_subject_ranges_for_splits(population: Sequence[str],
                                      proportion_train: float,
                                      proportion_test: float,
                                      proportion_val: float) \
            -> Dict[ModelExecutionMode, Set[str]]:
        """
        Get mutually exclusive subject ranges for each dataset split (w.r.t to the proportion provided)
        ensuring all sets have at least one item in them when possible.

        :param population: all subjects
        :param proportion_train: proportion for the train set.
        :param proportion_test: proportion for the test set.
        :param proportion_val: proportion for the validation set.
        :return: Train, Test, and Val splits
        """
        sum_proportions = proportion_train + proportion_val + proportion_test
        if not np.isclose(sum_proportions, 1):
            raise ValueError("proportion_train({}) + proportion_val({}) + proportion_test({}) must be ~ 1, found: {}"
                             .format(proportion_train, proportion_val, proportion_test, sum_proportions))

        if not 0 <= proportion_test < 1:
            raise ValueError("proportion_test({}) must be in range [0, 1)"
                             .format(proportion_test))

        if not all([0 < x < 1 for x in [proportion_train, proportion_val]]):
            raise ValueError("proportion_train({}) and proportion_val({}) must be in range (0, 1)"
                             .format(proportion_train, proportion_val))

        subjects_train, subjects_test, subjects_val = (set(population[0:1]),
                                                       set(population[1:2]),
                                                       set(population[2:3]))
        remaining = list(population[3:])
        if proportion_test == 0:
            remaining = list(subjects_test) + remaining
            subjects_test = set()

        subjects_train |= set(remaining[: ceil(len(remaining) * proportion_train)])
        if len(subjects_test) > 0:
            subjects_test |= set(remaining[len(subjects_train):
                                           len(subjects_train) + ceil(len(remaining) * proportion_test)])
        subjects_val |= set(remaining) - (subjects_train | subjects_test)
        result = {
            ModelExecutionMode.TRAIN: subjects_train,
            ModelExecutionMode.TEST: subjects_test,
            ModelExecutionMode.VAL: subjects_val
        }
        return result

    @staticmethod
    def _from_split_keys(df: pd.DataFrame,
                         train_keys: Sequence[str],
                         test_keys: Sequence[str],
                         val_keys: Sequence[str],
                         *,  # make column names keyword-only arguments to avoid mistakes when providing both
                         key_column: str,
                         subject_column: str,
                         group_column: Optional[str]) -> DatasetSplits:
        """
        Takes a slice of values from each data split train/test/val for the provided keys.

        :param df: the input DataFrame
        :param train_keys: keys for training.
        :param test_keys: keys for testing.
        :param val_keys: keys for validation.
        :param key_column: name of the column the provided keys belong to
        :param subject_column: subject id column name
        :param group_column: grouping column name; if given, samples from each group will always be
            in the same subset (train, val, or test) and cross-validation fold.
        :return: Data splits with respected dataset split ids.
        """
        train_df = DatasetSplits.get_df_from_ids(df, train_keys, key_column)
        test_df = DatasetSplits.get_df_from_ids(df, test_keys, key_column)
        val_df = DatasetSplits.get_df_from_ids(df, val_keys, key_column)

        return DatasetSplits(train=train_df, test=test_df, val=val_df,
                             subject_column=subject_column, group_column=group_column)

    @staticmethod
    def from_proportions(df: pd.DataFrame,
                         proportion_train: float,
                         proportion_test: float,
                         proportion_val: float,
                         *,  # make column names keyword-only arguments to avoid mistakes when providing both
                         subject_column: str = "",
                         group_column: Optional[str] = None,
                         shuffle: bool = True,
                         random_seed: int = 0) -> DatasetSplits:
        """
        Creates a split of a dataset into train, test, and validation set, according to fixed proportions using
        the "subject" column in the dataframe, or the group column, if given.

        :param df: The dataframe containing all subjects.
        :param proportion_train: proportion for the train set.
        :param proportion_test: proportion for the test set.
        :param subject_column: Subject id column name
        :param group_column: grouping column name; if given, samples from each group will always be
            in the same subset (train, val, or test) and cross-validation fold.
        :param proportion_val: proportion for the validation set.
        :param shuffle: If True the subjects in the dataframe will be shuffle before performing splits.
        :param random_seed: Random seed to be used for shuffle 0 is default.
        :return:
        """
        key_column: str = subject_column if group_column is None else group_column
        split_keys = df[key_column].unique()
        if shuffle:
            # fix the random seed so we can guarantee reproducibility when working with shuffle
            random.Random(random_seed).shuffle(split_keys)
        ranges = DatasetSplits.get_subject_ranges_for_splits(
            split_keys,
            proportion_train=proportion_train,
            proportion_val=proportion_val,
            proportion_test=proportion_test
        )
        return DatasetSplits._from_split_keys(df,
                                              list(ranges[ModelExecutionMode.TRAIN]),
                                              list(ranges[ModelExecutionMode.TEST]),
                                              list(ranges[ModelExecutionMode.VAL]),
                                              key_column=key_column,
                                              subject_column=subject_column,
                                              group_column=group_column)

    @staticmethod
    def from_subject_ids(df: pd.DataFrame,
                         train_ids: Sequence[str],
                         test_ids: Sequence[str],
                         val_ids: Sequence[str],
                         *,  # make column names keyword-only arguments to avoid mistakes when providing both
                         subject_column: str = "",
                         group_column: Optional[str] = None) -> DatasetSplits:
        """
        Assuming a DataFrame with columns subject
        Takes a slice of values from each data split train/test/val for the provided ids.

        :param df: the input DataFrame
        :param train_ids: ids for training.
        :param test_ids: ids for testing.
        :param val_ids: ids for validation.
        :param subject_column: subject id column name
        :param group_column: grouping column name; if given, samples from each group will always be
            in the same subset (train, val, or test) and cross-validation fold.
        :return: Data splits with respected dataset split ids.
        """
        return DatasetSplits._from_split_keys(df, train_ids, test_ids, val_ids, key_column=subject_column,
                                              subject_column=subject_column, group_column=group_column)

    @staticmethod
    def from_groups(df: pd.DataFrame,
                    train_groups: Sequence[str],
                    test_groups: Sequence[str],
                    val_groups: Sequence[str],
                    *,  # make column names keyword-only arguments to avoid mistakes when providing both
                    group_column: str,
                    subject_column: str = "") -> DatasetSplits:
        """
        Assuming a DataFrame with columns subject
        Takes a slice of values from each data split train/test/val for the provided groups.

        :param df: the input DataFrame
        :param train_groups: groups for training.
        :param test_groups: groups for testing.
        :param val_groups: groups for validation.
        :param subject_column: subject id column name
        :param group_column: grouping column name; if given, samples from each group will always be
            in the same subset (train, val, or test) and cross-validation fold.
        :return: Data splits with respected dataset split ids.
        """
        return DatasetSplits._from_split_keys(df, train_groups, test_groups, val_groups, key_column=group_column,
                                              subject_column=subject_column, group_column=group_column)

    @staticmethod
    def get_df_from_ids(df: pd.DataFrame, ids: Sequence[str],
                        subject_column: Optional[str] = "") -> pd.DataFrame:
        """
        Retrieve a subset dataframe where the subject column is restricted to a sequence of provided ids

        :param df: The dataframe to restrict
        :param ids: The ids to lookup
        :param subject_column: The column to lookup ids in. Defaults to ""
        :return: A subset of the dataframe
        """
        return df[df[subject_column].isin(ids)]

    def get_k_fold_cross_validation_splits(self, n_splits: int, random_seed: int = 0) -> List[DatasetSplits]:
        """
        Creates K folds from the Train + Val splits.
        If a group_column has been specified, the folds will be split such that
        subjects in a group will not be separated. In this case, the splits are
        fully deterministic, and random_seed is ignored.

        :param n_splits: number of folds to perform.
        :param random_seed: random seed to be used for shuffle 0 is default.
        :return: List of K dataset splits
        """
        if n_splits <= 0:
            raise ValueError("n_splits must be >= 0 found {}".format(n_splits))
        # concatenate train and val, as training set = train + val
        cv_dataset = pd.concat([self.train, self.val])
        if self.group_column is None:  # perform standard subject-based k-fold cross-validation
            # unique subjects
            subject_ids = cv_dataset[self.subject_column].unique()
            # calculate the random split indices
            k_folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            folds_gen = k_folds.split(subject_ids)
        else:  # perform grouped k-fold cross-validation
            # Here we take all entries, rather than unique, to keep subjects
            # matched to groups in the resulting arrays. This works, but could
            # perhaps be improved with group-by logic...?
            subject_ids = cv_dataset[self.subject_column].values
            groups = cv_dataset[self.group_column].values
            # scikit-learn uses a deterministic algorithm for grouped splits
            # that tries to balance the group sizes in all folds
            k_folds = GroupKFold(n_splits=n_splits)
            folds_gen = k_folds.split(subject_ids, groups=groups)

        def ids_from_indices(indices: Sequence[int]) -> List[str]:
            return [subject_ids[x] for x in indices]

        # create the number of requested splits of the dataset
        return [
            DatasetSplits(train=self.get_df_from_ids(cv_dataset, ids_from_indices(train_indices), self.subject_column),
                          val=self.get_df_from_ids(cv_dataset, ids_from_indices(val_indices), self.subject_column),
                          test=self.test, subject_column=self.subject_column, group_column=self.group_column)
            for train_indices, val_indices in folds_gen]
