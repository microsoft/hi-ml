#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

BIOMED_VLP_CXR_BERT_SPECIALIZED = "microsoft/BiomedVLP-CXR-BERT-specialized"
REPO_URL = f"https://huggingface.co/{BIOMED_VLP_CXR_BERT_SPECIALIZED}"
CXR_BERT_COMMIT_TAG = "v1.0"

BIOVIL_IMAGE_WEIGHTS_NAME = "biovil_image_resnet50_proj_size_128.pt"
BIOVIL_IMAGE_WEIGHTS_URL = f"{REPO_URL}/resolve/main/{BIOVIL_IMAGE_WEIGHTS_NAME}"
BIOVIL_IMAGE_WEIGHTS_MD5 = "02ce6ee460f72efd599295f440dbb453"


__all__ = [
    "BIOMED_VLP_CXR_BERT_SPECIALIZED",
    "REPO_URL",
    "CXR_BERT_COMMIT_TAG",
    "BIOVIL_IMAGE_WEIGHTS_NAME",
    "BIOVIL_IMAGE_WEIGHTS_URL",
    "BIOVIL_IMAGE_WEIGHTS_MD5",
]
