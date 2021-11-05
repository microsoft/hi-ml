#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# From:
# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train
# -hyperparameter-tune-deploy-with-sklearn/train_iris.py
import argparse
from pathlib import Path

import numpy as np
from azureml.core import ScriptRunConfig
from azureml.core.run import Run
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal, choice
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from health_azure import submit_to_azure_if_needed


def main() -> None:
    param_sampling = RandomParameterSampling({
        "--kernel": choice('linear', 'rbf', 'poly', 'sigmoid'),
        "--penalty": choice(0.5, 1, 1.5)
    })

    hyperdrive_config = HyperDriveConfig(
        run_config=ScriptRunConfig(source_directory=""),
        hyperparameter_sampling=param_sampling,
        primary_metric_name='Accuracy',
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=12,
        max_concurrent_runs=4)

    run_info = submit_to_azure_if_needed(
        compute_cluster_name="lite-testing-ds2",
        default_datastore="himldatasets",
        input_datasets=["himl_sample7_input"],
        wait_for_completion=True,
        wait_for_completion_show_output=True,
        hyperdrive_config=hyperdrive_config)
    if run_info.run is None:
        raise ValueError("run_info.run is None")
    run: Run = run_info.run
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='Penalty parameter of the error term')

    args = parser.parse_args()
    run.log('Kernel type', np.str(args.kernel))  # type: ignore
    run.log('Penalty', np.float(args.penalty))  # type: ignore

    # X -> features, y -> label
    input_folder = run_info.input_datasets[0] or Path("inputs")
    X = np.loadtxt(fname=input_folder / "X.csv", delimiter=',', skiprows=1)
    y = np.loadtxt(fname=input_folder / "y.csv", dtype='str', delimiter=',', skiprows=1)

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # training a linear SVM classifier
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel=args.kernel, C=args.penalty).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    # model accuracy for X_test
    accuracy = svm_model_linear.score(X_test, y_test)
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))  # type: ignore
    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    print(cm)


if __name__ == "__main__":
    main()
