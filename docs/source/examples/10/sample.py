#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# Inspired by:
# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train
# -hyperparameter-tune-deploy-with-sklearn/train_iris.py
import argparse
from pathlib import Path

import numpy as np
from azureml.core.run import Run

from health_azure import create_crossval_hyperdrive_config, submit_to_azure_if_needed


def main() -> None:
    num_cross_validation_splits = 2
    metric_name = "val/loss"
    hyperdrive_config = create_crossval_hyperdrive_config(num_cross_validation_splits,
                                                          cross_val_index_arg_name="cross_validation_split_index",
                                                          metric_name=metric_name)
    # tags{"num_splits": str(num_splits)}
    tags = {}
    run_info = submit_to_azure_if_needed(
        compute_cluster_name="lite-testing-ds2",
        default_datastore="himldatasets",
        input_datasets=["himl_kfold_split_iris"],
        wait_for_completion=True,
        conda_environment_file=Path(__file__).parent / "environment.yml",
        wait_for_completion_show_output=True,
        tags=tags,
        hyperdrive_config=hyperdrive_config,
        submit_to_azureml=True
    )

    if run_info.run is None:
        raise ValueError("run_info.run is None")
    run: Run = run_info.run

    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='Penalty parameter of the error term')
    parser.add_argument('--cross_validation_split_index', help="An index denoting which split of the dataset this"
                                                               "run represents in k-fold cross-validation")
    parser.add_argument("--num_splits", help="The total number of splits being used for k-fol cross validation")

    args = parser.parse_args()
    run.log('Kernel type', args.kernel)
    run.log('Penalty', args.penalty)

    # X -> features, y -> label
    input_folder = run_info.input_datasets[0] or Path("inputs")
    train_data_file = input_folder / "iris_data.csv"
    targets_file = input_folder / "iris_targets.csv"

    X = np.loadtxt(fname=train_data_file, delimiter=',').astype(float)
    y = np.loadtxt(fname=targets_file, dtype='str', delimiter=',').astype(float)

    # training a linear SVM classifier
    from sklearn.svm import SVC
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import LabelBinarizer

    # Parent run should perform the dataset split for k-fold cv
    train_splits_file = str(input_folder / "iris_data_splits.csv")
    test_splits_file = str(input_folder / "iris_target_splits.csv")

    train_splits_indices = np.loadtxt(fname=train_splits_file, delimiter=",").astype(int)
    test_splits_indices = np.loadtxt(fname=test_splits_file, delimiter=",").astype(int)

    fold = int(args.cross_validation_split_index)
    fold_train_idx = train_splits_indices[fold]
    fold_test_idx = test_splits_indices[fold]

    X_train, X_test = X[fold_train_idx], X[fold_test_idx]
    y_train, y_test = y[fold_train_idx], y[fold_test_idx]

    svm_model_linear = SVC(kernel=args.kernel, C=args.penalty).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)
    lb = LabelBinarizer()
    y_pred = lb.fit_transform(svm_predictions)

    # model accuracy for X_test
    loss = log_loss(y_test, y_pred)

    print(f"Loss for fold {fold}: {loss}")
    # log val/loss
    run.log(metric_name, loss)


if __name__ == "__main__":
    main()
