#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

BIOMED_VLP_CXR_BERT_SPECIALIZED = "microsoft/BiomedVLP-CXR-BERT-specialized"
REPO_URL = f"https://huggingface.co/{BIOMED_VLP_CXR_BERT_SPECIALIZED}"
CXR_BERT_COMMIT_TAG = "v1.0"

IMAGE_WEIGHTS_NAME = ""  # TODO
IMAGE_WEIGHTS_URL = f"{REPO_URL}/raw/main/{IMAGE_WEIGHTS_NAME}"
IMAGE_WEIGHTS_MD5 = ""  # TODO


__all__ = [
    "BIOMED_VLP_CXR_BERT_SPECIALIZED",
    "REPO_URL",
    "CXR_BERT_COMMIT_TAG",
    "IMAGE_WEIGHTS_NAME",
    "IMAGE_WEIGHTS_URL",
    "IMAGE_WEIGHTS_MD5",
]
