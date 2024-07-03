#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Simple 'hello world' script to elevate to AML using our `submit_to_azure_if_needed` function.

Invoke like this:
    python hello_world.py --cluster <name_of_compute_cluster>

"""
import os
import sys
from argparse import ArgumentParser
from typing import Callable, Union
from pathlib import Path
from datetime import datetime

from azure.identity import get_bearer_token_provider, AzureCliCredential, ManagedIdentityCredential
import openai

# Add hi-ml packages to sys.path so that AML can find them
himl_azure_root = Path(__file__).resolve().parent
folders_to_add = [himl_azure_root / "src"]
for folder in folders_to_add:
    if folder.is_dir():
        sys.path.insert(0, str(folder))

from health_azure import submit_to_azure_if_needed, DatasetConfig, is_running_in_azure_ml
from health_azure.logging import logging_to_stdout

# The default scope for the Azure Cognitive Services. Tokens are retrieve from this page, and later used instead
# of the API key.
AZURE_COGNITIVE_SERVICES = "https://cognitiveservices.azure.com"

ENV_AZUREML_IDENTITY_ID = "DEFAULT_IDENTITY_CLIENT_ID"


def get_credential() -> Union[AzureCliCredential, ManagedIdentityCredential]:
    """Get the appropriate Azure credential based on the environment. The credential is a managed identity when running
    in AzureML, otherwise Azure CLI credential."""
    if is_running_in_azure_ml():
        return ManagedIdentityCredential(client_id=os.environ[ENV_AZUREML_IDENTITY_ID])
    return AzureCliCredential()


def get_azure_token_provider() -> Callable[[], str]:
    """Get a token provider for Azure Cognitive Services. The bearer token provider gets authentication tokens and
    refreshes them automatically upon expiry.
    """
    credential = get_credential()
    credential.get_token(AZURE_COGNITIVE_SERVICES)
    return get_bearer_token_provider(credential, AZURE_COGNITIVE_SERVICES)


def main() -> None:
    """
    Write out the given message, in an AzureML 'experiment' if required.

    First call submit_to_azure_if_needed.
    """
    parser = ArgumentParser()
    parser.add_argument("-c", "--cluster", type=str, required=True, help="The name of the compute cluster to run on")
    parser.add_argument("--openai_url", type=str, required=False, help="The URL of the OpenAI endpoint to use")
    parser.add_argument("--openai_model", type=str, required=False, help="The OpenAI deployment to use")
    parser.add_argument("--input_dataset", type=str, required=False, help="The name of the input dataset to enumerate")
    parser.add_argument("--output_dataset", type=str, required=False, help="The name of the output dataset")
    args = parser.parse_args()
    input_datasets: list[DatasetConfig] = []
    output_datasets: list[DatasetConfig] = []
    if args.input_dataset:
        input_datasets.append(DatasetConfig(name=args.input_dataset, use_mounting=True))
    if args.output_dataset:
        output_datasets.append(DatasetConfig(name=args.output_dataset, use_mounting=True))
    logging_to_stdout
    run_info = submit_to_azure_if_needed(
        compute_cluster_name=args.cluster,
        strictly_aml_v1=True,
        submit_to_azureml=True,
        workspace_config_file=himl_azure_root / "config.json",
        snapshot_root_directory=himl_azure_root,
        input_datasets=input_datasets,
        output_datasets=output_datasets,
        conda_environment_file=himl_azure_root / "environment_hello_world.yml",
    )
    print("Hello Chris! This is your first successful AzureML run :-)")
    any_error = False
    if args.input_dataset:
        try:
            input_dataset = run_info.input_datasets[0]
            assert input_dataset is not None
            if input_dataset.is_file():
                print(f"Input dataset is a single file: {input_dataset}")
            elif input_dataset.is_dir():
                print(f"Files in input dataset folder {input_dataset}:")
                for file in input_dataset.glob("*"):
                    print(file)
            else:
                print(f"Input dataset is neither a file nor a folder: {input_dataset}")
                any_error = True
        except Exception as e:
            print(f"Failed to read input dataset: {e}")
            any_error = True
    else:
        print("No input dataset was specified.")
    if args.output_dataset:
        try:
            output_dataset = run_info.output_datasets[0]
            assert output_dataset is not None
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
            output_file = output_dataset / f"hello_world_{timestamp}.txt"
            output_text = f"Calling all dentists, the private song group starts at {timestamp}!"
            print(f"Writing to output file: {output_text}")
            output_file.write_text(output_text)
        except Exception as e:
            print(f"Failed to write output dataset: {e}")
            any_error = True
    else:
        print("No output dataset was specified..")
    if args.openai_url and args.openai_model:
        try:
            print(f"OpenAI URL: {args.openai_url}")
            token_provider = get_azure_token_provider()
            openai.api_version = "2023-12-01-preview"
            openai.azure_endpoint = args.openai_url
            openai.azure_ad_token_provider = token_provider
            prompt = "Write a 4 line poem using the words 'private', 'dentist', 'song' and 'group'. "
            print(f"Prompt: {prompt}")
            response = openai.chat.completions.create(
                model=args.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.8,
            )
            content = response.choices[0].message.content
            print(f"Response from OpenAI: {content}")
        except Exception as e:
            print(f"Failed to connect to OpenAI: {e}")
            any_error = True

    if any_error:
        raise RuntimeError("There were errors during the run.")


if __name__ == "__main__":
    main()
