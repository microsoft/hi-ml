#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pickle
from io import BytesIO
from typing import Any, Optional, Callable, Dict, Union

import torch

from health_azure import RUN_CONTEXT, object_to_yaml, is_running_in_azure_ml


def _dump_to_stream(o: Any) -> BytesIO:
    """
    Writes the given object to a byte stream in pickle format, and returns the stream.

    :param o: The object to pickle.
    :return: A byte stream, with the position set to 0.
    """
    stream = BytesIO()
    pickle.dump(o, file=stream)
    stream.seek(0)
    return stream


class ModelInfo:
    """Stores a model, its example input, and metadata that describes how the model was trained."""

    MODEL = "model"
    MODEL_EXAMPLE_INPUT = "model_example_input"
    MODEL_CONFIG_YAML = "model_config_yaml"
    GIT_REPOSITORY = "git_repository"
    GIT_COMMIT_HASH = "git_commit_hash"
    DATASET_NAME = "dataset_name"
    AZURE_ML_WORKSPACE = "azure_ml_workspace"
    AZURE_ML_RUN_ID = "azure_ml_run_id"
    TEXT_TOKENIZER = "text_tokenizer"
    IMAGE_PRE_PROCESSING = "image_pre_processing"
    IMAGE_DIMENSIONS = "image_dimensions"
    OTHER_INFO = "other_info"
    OTHER_DESCRIPTION = "other_description"

    def __init__(
        self,
        model: Optional[Union[torch.nn.Module, torch.jit.ScriptModule]] = None,
        model_example_input: Optional[torch.Tensor] = None,
        model_config: Any = None,
        git_repository: str = "",
        git_commit_hash: str = "",
        dataset_name: str = "",
        azure_ml_workspace: str = "",
        azure_ml_run_id: str = "",
        text_tokenizer: Any = None,
        image_pre_processing: Optional[Callable] = None,
        image_dimensions: str = "",
        other_info: Any = None,
        other_description: str = "",
    ):
        """
        :param model: The model that should be serialized, or the deserialized model, defaults to None
        :param model_example_input: A tensor that can be input to the forward pass of the model, defaults to None
        :param model_config: The configuration object that was used to start the training run, defaults to None
        :param git_repository: The name of the git repository that contains the training codebase, defaults to ""
        :param git_commit_hash: The git commit hash that was used to run the training, defaults to ""
        :param dataset_name: The name of the dataset that was used to train the model, defaults to ""
        :param azure_ml_workspace: The name of the AzureML workspace that contains the training run, defaults to ""
        :param azure_ml_run_id: The AzureML run that did the training, defaults to ""
        :param text_tokenizer: A text tokenizer object to pre-process the model input (default: None). The object given
            here will be pickled before it is passed to ``torch.save``.
        :param image_pre_processing: An object that describes the processing for the image before it is input to the
            model, defaults to None
        :param image_dimensions: The size of the pre-processed image that is accepted by the model, defaults to ""
        :param other_info: An arbitray object that will also be written to the checkpoint. For example, this can be a
            binary stream. The object provided here will be pickled before it is passed to ``torch.save``.
        :param other_description: A human-readable description of what the ``other_info`` field contains.
        """
        self.model = model
        self.model_example_input = model_example_input
        self.model_config = model_config
        self.git_repository = git_repository
        self.git_commit_hash = git_commit_hash
        self.dataset_name = dataset_name
        self.azure_ml_workspace = azure_ml_workspace
        self.azure_ml_run_id = azure_ml_run_id
        self.text_tokenizer = text_tokenizer
        self.image_pre_processing = image_pre_processing
        self.image_dimensions = image_dimensions
        self.other_info = other_info
        self.other_description = other_description

    def get_metadata_from_azureml(self) -> None:
        """Reads information about the git repository and AzureML-related info from the AzureML run context.
        If any of those information are already stored in the object, those have higher priority.
        """
        if not is_running_in_azure_ml():
            return
        if not self.azure_ml_workspace:
            self.azure_ml_workspace = RUN_CONTEXT.experiment.workspace.name
        if not self.azure_ml_run_id:
            self.azure_ml_run_id = RUN_CONTEXT.id
        properties: Dict = RUN_CONTEXT.get_properties()
        if not self.git_repository:
            self.git_repository = properties.get("azureml.git.repository_uri", "")
        if not self.git_commit_hash:
            self.git_commit_hash = properties.get("azureml.git.commit", "")

    def state_dict(self, strict: bool = True) -> Dict[str, Any]:
        """Creates a dictionary representation of the current object.

        :param strict: The setting for the 'strict' flag in the call to torch.jit.trace.
        """

        def bytes_or_none(o: Any) -> Optional[bytes]:
            return _dump_to_stream(o).getvalue() if o is not None else None

        if self.model is None or self.model_example_input is None:
            raise ValueError("To generate a state dict, the model and model_example_input must be present.")
        try:
            traced_model = torch.jit.trace(self.model, self.model_example_input, strict=strict)
        except Exception as ex:
            raise ValueError(f"Unable to convert the model to torchscript: {ex}")
        jit_stream = BytesIO()
        torch.jit.save(traced_model, jit_stream)
        try:
            config_yaml = object_to_yaml(self.model_config) if self.model_config is not None else None
        except Exception as ex:
            raise ValueError(f"Unable to convert the model configuration to YAML: {ex}")
        return {
            ModelInfo.MODEL: jit_stream.getvalue(),
            ModelInfo.MODEL_EXAMPLE_INPUT: self.model_example_input,
            ModelInfo.MODEL_CONFIG_YAML: config_yaml,
            ModelInfo.GIT_REPOSITORY: self.git_repository,
            ModelInfo.GIT_COMMIT_HASH: self.git_commit_hash,
            ModelInfo.DATASET_NAME: self.dataset_name,
            ModelInfo.AZURE_ML_WORKSPACE: self.azure_ml_workspace,
            ModelInfo.AZURE_ML_RUN_ID: self.azure_ml_run_id,
            ModelInfo.TEXT_TOKENIZER: bytes_or_none(self.text_tokenizer),
            ModelInfo.IMAGE_PRE_PROCESSING: bytes_or_none(self.image_pre_processing),
            ModelInfo.IMAGE_DIMENSIONS: self.image_dimensions,
            ModelInfo.OTHER_INFO: bytes_or_none(self.other_info),
            ModelInfo.OTHER_DESCRIPTION: self.other_description,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads a dictionary representation into the current object, overwriting all matching fields.

        :param state_dict: The dictionary to load from.
        """

        def unpickle_from_bytes(field: str) -> Any:
            if field not in state_dict:
                raise KeyError(f"State_dict does not contain a field '{field}'")
            b = state_dict[field]
            if b is None:
                return None
            stream = BytesIO(b)
            try:
                o = pickle.load(stream)
            except Exception as ex:
                raise ValueError(f"Failure when unpickling field '{field}': {ex}")
            return o

        self.model = torch.jit.load(BytesIO(state_dict[ModelInfo.MODEL]))
        self.model_example_input = state_dict[ModelInfo.MODEL_EXAMPLE_INPUT]
        self.model_config = state_dict[ModelInfo.MODEL_CONFIG_YAML]
        self.git_repository = state_dict[ModelInfo.GIT_REPOSITORY]
        self.git_commit_hash = state_dict[ModelInfo.GIT_COMMIT_HASH]
        self.dataset_name = state_dict[ModelInfo.DATASET_NAME]
        self.azure_ml_workspace = state_dict[ModelInfo.AZURE_ML_WORKSPACE]
        self.azure_ml_run_id = state_dict[ModelInfo.AZURE_ML_RUN_ID]
        self.text_tokenizer = unpickle_from_bytes(ModelInfo.TEXT_TOKENIZER)
        self.image_pre_processing = unpickle_from_bytes(ModelInfo.IMAGE_PRE_PROCESSING)
        self.image_dimensions = state_dict[ModelInfo.IMAGE_DIMENSIONS]
        self.other_info = unpickle_from_bytes(ModelInfo.OTHER_INFO)
        self.other_description = state_dict[ModelInfo.OTHER_DESCRIPTION]
