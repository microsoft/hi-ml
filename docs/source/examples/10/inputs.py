#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
from azureml.core import Datastore
from sklearn import datasets
from sklearn.model_selection import KFold

from health_azure import get_workspace
from health_azure.datasets import get_datastore


def main() -> None:
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    inputs = Path("inputs")
    inputs.mkdir(exist_ok=False)
    train_data_file = inputs / "iris_data.csv"
    np.savetxt(train_data_file, X, delimiter=',')
    targets_file = inputs / "iris_targets.csv"
    np.savetxt(targets_file, y, delimiter=',')

    print("Creating splits")
    num_cross_validation_splits = 5
    k_folds = KFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=0)
    splits = np.array(list(k_folds.split(X)))
    indices_train_splits, indices_test_splits = [], []
    for split in splits:
        indices_train_splits.append(split[0])
        indices_test_splits.append(split[1])

    train_splits_file = inputs / "iris_data_splits.csv"
    target_splits_file = inputs / "iris_target_splits.csv"

    np.savetxt(str(train_splits_file), np.vstack(indices_train_splits), delimiter=",")
    np.savetxt(str(target_splits_file), np.vstack(indices_test_splits), delimiter=",")

    ws = get_workspace()
    datastore: Datastore = get_datastore(workspace=ws,
                                         datastore_name="himldatasets")

    dataset_name = 'himl_kfold_split_iris'
    datastore.upload_files(
        [str(train_data_file), str(targets_file), str(train_splits_file), str(target_splits_file)],
        relative_root=str(inputs),
        target_path=dataset_name,
        overwrite=True,
        show_progress=True)

    train_data_file.unlink()
    targets_file.unlink()
    train_splits_file.unlink()
    target_splits_file.unlink()
    inputs.rmdir()


if __name__ == "__main__":
    main()
