#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
from sklearn import datasets

import health.azure.himl as himl
from health.azure.datasets import get_datastore


def main() -> None:
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    dataset = Path("dataset")
    dataset.mkdir(exist_ok=False)
    X_csv = dataset / "X.csv"
    np.savetxt(X_csv, X, delimiter=',')
    y_csv = dataset / "y.csv"
    np.savetxt(y_csv, y, delimiter=',')

    path = Path(__file__).parent.resolve()

    workspace = himl.get_workspace(aml_workspace=None,
                                   workspace_config_path=path / himl.WORKSPACE_CONFIG_JSON)

    datastore = get_datastore(workspace=workspace,
                              datastore_name="himldatasets")

    datastore.upload_files(
        [str(X_csv), str(y_csv)],
        relative_root=str(dataset),
        target_path='himl_sample7_input',
        overwrite=True,
        show_progress=True)

    X_csv.unlink()
    y_csv.unlink()
    dataset.rmdir()


if __name__ == "__main__":
    main()
