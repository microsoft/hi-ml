#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import re

from azureml._restclient.constants import RunStatus
from azureml.core import Experiment, Run, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication


def cancel_running_and_queued_jobs() -> None:
    print("Authenticating")
    auth = ServicePrincipalAuthentication(
        tenant_id='72f988bf-86f1-41af-91ab-2d7cd011db47',
        service_principal_id=os.environ["HIML_SERVICE_PRINCIPAL_ID"],
        service_principal_password=os.environ["HIML_SERVICE_PRINCIPAL_PASSWORD"])
    print("Getting AML workspace")
    workspace = Workspace.get(
        name="hi-ml",
        auth=auth,
        subscription_id=os.environ["HIML_SUBSCRIPTION_ID"],
        resource_group=os.environ["HIML_RESOURCE_GROUP"]
    )
    experiment_name = os.environ["HIML_EXPERIMENT_NAME"]
    experiment_name = re.sub("_+", "_", re.sub(r"\W+", "_", experiment_name))
    print(f"Experiment: {experiment_name}")
    experiment = Experiment(workspace, name=experiment_name)
    print(f"Retrieved experiment {experiment.name}")
    for run in experiment.get_runs(include_children=True, properties={}):
        assert isinstance(run, Run)
        status_suffix = f"'{run.status}' run {run.id} ({run.display_name})"
        if run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.FINALIZING, RunStatus.CANCELED,
                          RunStatus.CANCEL_REQUESTED):
            print(f"Skipping {status_suffix}")
        else:
            print(f"Cancelling {status_suffix}")
            try:
                run.cancel()
            except Exception as ex:
                # Exceptions here are rare, but do happen. Sometimes AML says "Run is in a terminal state", and
                # refuses to cancel.
                print(f"Unable to cancel {status_suffix}: {ex}")


if __name__ == "__main__":
    cancel_running_and_queued_jobs()
