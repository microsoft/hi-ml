#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import numpy as np
import pandas as pd
from health_cpath.models.transforms import PathOrString
from health_cpath.utils.naming import ModelKey
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Dataset
from typing import Any, Generic, Optional, Tuple, TypeVar, Union, List


StratifyType = Union[str, List[str]]
_DatasetT = TypeVar("_DatasetT", bound=Dataset)


def get_train_test_split_col(index: pd.Index, test_index: Union[np.ndarray, pd.Index]) -> pd.Series:
    """Create a series of 'train' and 'test' labels for the input index.

    :param index: Input index to be split.
    :param test_index: The index of the test set.
    :return: A series of 'train' and 'test' labels for the input index.
    """
    split_col = pd.Series(ModelKey.TRAIN, index=index)
    split_col[test_index] = ModelKey.TEST
    return split_col


def ungroup_series(grouped_series: pd.Series, groups: pd.Series) -> pd.Series:
    """Expand a grouped series by indexing it by a series of group labels.

    :param grouped_series: Series of values whose index corresponds to unique group labels.
    :param groups: Series of group labels indexed by individual elements.
    :return: A series with the same index as `groups` and values taken from `grouped_series`.
        The output values for all entries in a group are equal to the original grouped value.
    """
    ungrouped_series = grouped_series[groups]
    ungrouped_series.index = groups.index
    split_groupby = ungrouped_series.groupby(groups)
    assert (split_groupby.nunique() == 1).all(), "Splits are not unique within each group"
    assert split_groupby.first().equals(grouped_series), "Individual splits are inconsistent with group splits"
    return ungrouped_series


def split_index(
    index: pd.Index,
    test_frac: float,
    strata: Optional[Union[pd.Series, pd.DataFrame]],
    seed: Optional[int] = None
) -> pd.Series:
    """Assign each item of an index to random 'train' or 'test' split.

    :param index: Input index to be split.
    :param test_frac: Fraction of the dataset to reserve for testing, between 0 and 1.
    :param strata: One or more series to use for stratification (assumed containing discrete
        values), such that their joint distributions will be split approximately equally. If
        omitted, no stratification is performed. Their index should match the main splitting index.
    :param seed: The random seed to use in generating the splits, for reproducibility. If `None`
        (default), will produce different random splits every time by default.
    :return: A series of 'train' and 'test' labels for the input index.
    """
    _, test_index = train_test_split(index, test_size=test_frac, stratify=strata,
                                     shuffle=True, random_state=seed)
    assert isinstance(test_index, pd.Index)
    return get_train_test_split_col(index, test_index)


def split_dataframe(df: pd.DataFrame, test_frac: float, stratify: StratifyType = '',
                    group: str = '', seed: Optional[int] = None) -> pd.Series:
    """Assign each row of a dataframe to random 'train' or 'test' split.

    :param df: Input dataframe.
    :param test_frac: Fraction of the dataset to reserve for testing, between 0 and 1.
    :param stratify: One or more column names to use for stratification (assumed containing
        discrete values), such that their joint distributions will be split approximately equally.
        If omitted, no stratification is performed.
    :param group: A column name to use for grouping (e.g. subject ID), to prevent data leakage.
    :param seed: The random seed to use in generating the splits, for reproducibility.
        If `None` (default), will produce different random splits every time by default.
    :return: A series of 'train' and 'test' labels, with the same index as the input dataframe.
    """
    orig_df = df
    if group:
        groupby = orig_df.groupby(group)
        if stratify:
            assert (groupby[stratify].nunique() == 1).all(axis=None), "Strata are not unique within each group"
        df = groupby.first()

    index = df.index
    strata = df[stratify] if stratify else None

    split_col = split_index(index, test_frac=test_frac, strata=strata, seed=seed)

    if group:
        split_col = ungroup_series(split_col, groups=orig_df[group])

    return split_col


def split_dataframe_k_fold(
    df: pd.DataFrame,
    crossval_count: int = 1,
    stratify: StratifyType = '',
    group: str = '',
    seed: Optional[int] = 1,
) -> pd.Series:
    """Assign each row of a dataframe to a k-fold cross-validation split.

    :param df: Input dataframe.
    :param crossval_count: Number of folds.
    :param stratify: A column name to use for stratification (assumed containing discrete values).
    :param group: A column name to use for grouping (e.g. subject ID), to prevent data leakage.
    :param seed: The random seed to use in generating the splits, for reproducibility.
        If `None` (default), will produce different random splits every time by default.
    :return: A list of series of fold labels, with the same index as the input dataframe.
    """
    # Extract the stratification column (e.g. label) from the dataframe and the group column (e.g. case id)
    stratify_column = df[stratify] if stratify else None
    if isinstance(stratify, List):
        # StratifiedGroupKFold only accepts a single stratification column, so we need to combine them into one
        # This is done by concatenating the values of the columns into a single string
        stratify_column = stratify_column.apply(lambda row: '_'.join([str(each) for each in row]), axis=1)
    groups = df[group] if group else None

    # If there is no group column, use StratifiedKFold, StratifiedGroupKFold returns an empty test set if
    # there is nothig to groupby
    k_fold_class = StratifiedKFold if groups is None else StratifiedGroupKFold
    k_fold_generator = k_fold_class(n_splits=crossval_count, shuffle=True, random_state=seed)

    splits = []

    for train_index, test_index in k_fold_generator.split(X=df, y=stratify_column, groups=groups):
        if groups is not None:
            train_groups = set(groups[train_index])
            test_groups = set(groups[test_index])
            assert train_groups.isdisjoint(test_groups), "Train and test groups are not disjoint"
            assert train_groups.union(test_groups) == set(groups), "Missing groups in train or test split"
        splits.append(get_train_test_split_col(df.index, test_index))

    return splits


def split_dataframe_using_splits_csv(
    dataset_df: pd.DataFrame, splits_csv: PathOrString, index_col: str, train: bool
) -> pd.DataFrame:
    """Assign each row of a dataframe to a split based on a CSV file.

    :param dataset_df: Input dataframe that will be modified in-place.
    :param splits_csv: Path to a CSV file containing a column with the same name as `index_col`
    :param index_col: The name of the column in `dataset_df` that will be used to match rows to the splits CSV file.
    :param train: If `True`, the 'train' column of the splits CSV will be used. Otherwise, the 'test' column will
    be used.
    """
    splits_series = pd.read_csv(splits_csv, index_col=index_col).squeeze('columns')
    assert isinstance(splits_series, pd.Series)  # should be a single column for indexing, not a dataframe
    split = ModelKey.TRAIN if train else ModelKey.TEST
    dataset_df = dataset_df.loc[splits_series == split]
    return dataset_df


def split_tiles_dataframe_using_splits_csv(
    dataset_df: pd.DataFrame, splits_csv: PathOrString, slide_ids: pd.Series, index_col: str, train: bool
) -> pd.DataFrame:
    """Assign each row of a dataframe to a split based on a CSV file.

    :param dataset_df: Input dataframe that will be modified in-place.
    :param splits_csv: Path to a CSV file containing a column with the same name as `index_col`
    :param index_col: The name of the column in `dataset_df` that will be used to match rows to the splits CSV file.
    :param slide_ids: List of slide ids (instead of tile ids that are the index of the dataset_df)
    :param train: If `True`, the 'train' column of the splits CSV will be used. Otherwise, the 'test' column will
    be used.
    """
    splits_series = pd.read_csv(splits_csv, index_col=index_col).squeeze('columns')
    assert isinstance(splits_series, pd.Series)  # should be a single column for indexing, not a dataframe
    split = ModelKey.TRAIN if train else ModelKey.TEST
    tiles_selection = slide_ids.map(splits_series == split)
    dataset_df = dataset_df[tiles_selection]

    return dataset_df


class SplitsMixin(Generic[_DatasetT]):

    def __init__(
        self,
        crossval_count: int = 1,
        crossval_index: int = 0,
        stratify_by: StratifyType = '',
        groupd_by: str = '',
        **kwargs: Any,
    ) -> None:
        self.crossval_count = crossval_count
        self.crossval_index = crossval_index
        self.ssl_seed = 1  # This is hardcoded here to match SSL random seed, should be changed in the future to seed
        self.stratify_by = stratify_by
        self.groupd_by = groupd_by
        super().__init__(  # type: ignore
            crossval_count=crossval_count, crossval_index=crossval_index, **kwargs
        )

    def _get_dataset(self, dataset_df: Optional[pd.DataFrame] = None, train: Optional[bool] = None) -> _DatasetT:
        raise NotImplementedError

    def get_splits(self) -> Tuple[_DatasetT, _DatasetT, _DatasetT]:
        train_val_dataset = self._get_dataset(train=True)
        df = train_val_dataset.dataset_df
        assert isinstance(df, pd.DataFrame)  # for type-checking
        stratify_by = self.stratify_by or [train_val_dataset.label_column]
        assert set(stratify_by).issubset(set(df.columns)), f"stratify_by={stratify_by} not in df.columns={df.columns}"

        if self.crossval_count > 1:
            splits = split_dataframe_k_fold(
                df=df,
                crossval_count=self.crossval_count,
                stratify=stratify_by,
                group=self.groupd_by,
                seed=self.ssl_seed,
            )[self.crossval_index]
        else:
            try:
                splits = split_dataframe(
                    df,
                    test_frac=0.1,  # 10% of the data is used for validation as in SSL
                    stratify=stratify_by,
                    group=self.groupd_by,
                    seed=self.ssl_seed,
                )
            except ValueError as e:
                logging.warning(f"Failed to stratify by {stratify_by}: {e}")
                logging.warning(f"Falling back to stratification by label {train_val_dataset.label_column}")
                splits = split_dataframe(
                    df,
                    test_frac=0.1,  # 10% of the data is used for validation as in SSL
                    stratify=train_val_dataset.label_column,
                    group=self.groupd_by,
                    seed=self.ssl_seed,
                )

        assert splits.index.equals(df.index)
        train_indices: np.ndarray = np.where(splits == ModelKey.TRAIN)[0]
        val_indices: np.ndarray = np.where(splits == ModelKey.TEST)[0]

        return (
            self._get_dataset(dataset_df=df.iloc[train_indices]),
            self._get_dataset(dataset_df=df.iloc[val_indices]),
            self._get_dataset(train=False),
        )
