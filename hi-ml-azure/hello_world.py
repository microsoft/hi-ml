#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Simple 'hello world' script to elevate to AML using our `submit_to_azure_if_needed` function.

Invoke like this:
    python hello_world.py --cluster <name_of_compute_cluster>

"""
import sys
from argparse import ArgumentParser
from typing import Callable
from pathlib import Path

from azure.identity import get_bearer_token_provider
import openai

# Add hi-ml packages to sys.path so that AML can find them
himl_azure_root = Path(__file__).resolve().parent
folders_to_add = [himl_azure_root / "src"]
for folder in folders_to_add:
    if folder.is_dir():
        sys.path.insert(0, str(folder))

from health_azure import submit_to_azure_if_needed, DatasetConfig
from health_azure.logging import logging_to_stdout
from health_azure.auth import get_credential

# The default scope for the Azure Cognitive Services. Tokens are retrieve from this page, and later used instead
# of the API key.
AZURE_COGNITIVE_SERVICES = "https://cognitiveservices.azure.com"


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
    parser.add_argument("--dataset", type=str, required=False, help="The name of the dataset to mount and enumerate")
    args = parser.parse_args()
    input_datasets: list[DatasetConfig] = []
    DATASET_FOLDER = Path("/dataset")
    if args.dataset:
        input_datasets.append(DatasetConfig(name=args.dataset, target_folder=DATASET_FOLDER, use_mounting=True))
    logging_to_stdout
    # _ = submit_to_azure_if_needed(
    #     compute_cluster_name=args.cluster,
    #     strictly_aml_v1=True,
    #     submit_to_azureml=True,
    #     workspace_config_file=himl_azure_root / "config.json",
    #     snapshot_root_directory=himl_azure_root,
    #     input_datasets=input_datasets,
    # )
    print("Hello Chris! This is your first successful AzureML run :-)")
    if args.dataset:
        print(f"Dataset {args.dataset} was mounted at {DATASET_FOLDER}")
        print("Files in the dataset:")
        for file in DATASET_FOLDER.glob("*"):
            print(file)
    else:
        print("No dataset was mounted.")
    if args.openai_url and args.openai_model:
        print(f"OpenAI URL: {args.openai_url}")
        token_provider = get_azure_token_provider()
        openai.api_version = "2023-12-01-preview"
        openai.azure_endpoint = args.openai_url
        openai.azure_ad_token_provider = token_provider
        try:
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
            print(f"Failed to connect to openai: {e}")
            raise


if __name__ == "__main__":
    main()
