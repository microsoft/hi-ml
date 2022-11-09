#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
from azureml.core import Datastore
from sklearn import datasets

from health_azure import get_workspace
from health_azure.datasets import get_datastore


def main() -> None:
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    inputs = Path("inputs")
    inputs.mkdir(exist_ok=False)
    X_csv = inputs / "X.csv"
    np.savetxt(X_csv, X, delimiter=',')
    y_csv = inputs / "y.csv"
    np.savetxt(y_csv, y, delimiter=',')

    workspace = get_workspace()

    datastore: Datastore = get_datastore(workspace=workspace,
                                         datastore_name="himldatasets")

    datastore.upload_files(
        [str(X_csv), str(y_csv)],
        relative_root=str(inputs),
        target_path='himl_sample7_input',
        overwrite=True,
        show_progress=True)

    X_csv.unlink()
    y_csv.unlink()
    inputs.rmdir()


if __name__ == "__main__":
    main()
