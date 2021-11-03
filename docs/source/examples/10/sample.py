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

ROOT_DIR = Path.cwd() /"hi-ml-azure"/ "src"
import sys
sys.path.append(str(ROOT_DIR))
from health_azure import submit_to_azure_if_needed


def main() -> None:

    run_info = submit_to_azure_if_needed(
        snapshot_root_directory=ROOT_DIR,
        compute_cluster_name="lite-testing-ds2",
        default_datastore="himldatasets",
        input_datasets=["himl_sample7_input"],
        wait_for_completion=True,
        conda_environment_file=Path(__file__).parent / "environment.yml",
        wait_for_completion_show_output=True,
        num_cross_validation_splits=2
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
    parser.add_argument("--num_cross_validation_splits", help="The total number of splits being used for k-fold"
                                                              "cross validation")

    args = parser.parse_args()
    run.log('Kernel type', args.kernel)
    run.log('Penalty', args.penalty)

    # X -> features, y -> label
    input_folder = Path(__file__).parent / "inputs"
    X = np.loadtxt(fname=input_folder / "X.csv", delimiter=',').astype(float)
    y = np.loadtxt(fname=input_folder / "y.csv", dtype='str', delimiter=',').astype(float)

    # training a linear SVM classifier
    from sklearn.svm import SVC
    from sklearn.metrics import log_loss
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelBinarizer

    # Parent run should perform the dataset split for k-fold cv
    train_splits_file = str(input_folder / "x_splits.csv")
    test_splits_file = str(input_folder / "y_splits.csv")

    # if we aren't running inside a child run, create the dataset if it doesn't already exist
    if args.cross_validation_split_index is None and not Path(train_splits_file).is_file():
        print("Creating splits")
        k_folds = KFold(n_splits=int(args.num_cross_validation_splits), shuffle=True, random_state=0)
        splits = np.array(list(k_folds.split(X)))
        indices_train_splits, indices_test_splits = [], []
        for split in splits:
            indices_train_splits.append(split[0])
            indices_test_splits.append(split[1])
        np.savetxt(train_splits_file, np.vstack(indices_train_splits), delimiter=",")
        np.savetxt(test_splits_file, np.vstack(indices_test_splits), delimiter=",")

    else:
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
        run.log('val/loss', loss)


if __name__ == "__main__":
    main()
