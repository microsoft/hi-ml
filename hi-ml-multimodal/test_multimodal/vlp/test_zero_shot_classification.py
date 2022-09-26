import tempfile
from enum import Enum, unique
from pathlib import Path
from typing import List, Tuple, Union

import requests
from torchvision.datasets.utils import check_integrity

from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.model.model import get_biovil_resnet
from health_multimodal.text.utils import get_cxr_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine

RESIZE = 512
CENTER_CROP_SIZE = 512


@unique
class ClassType(str, Enum):
    """Enum for the different types of CXR abnormality classes."""
    PNEUMONIA = "pneumonia"
    NO_PNEUMONIA = "no_pneumonia"


def _get_vlp_inference_engine() -> ImageTextInferenceEngine:
    image_inference = ImageInferenceEngine(
        image_model=get_biovil_resnet(pretrained=True),
        transform=create_chest_xray_transform_for_inference(resize=RESIZE, center_crop_size=CENTER_CROP_SIZE))
    img_txt_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=get_cxr_bert_inference(),
    )
    return img_txt_inference


def _get_default_text_prompts_for_pneumonia() -> Tuple[List, List]:
    """
    Get the default text prompts for presence and absence of pneumonia
    """
    pos_query = ['Findings consistent with pneumonia', 'Findings suggesting pneumonia',
                 'This opacity can represent pneumonia', 'Findings are most compatible with pneumonia']
    neg_query = ['There is no pneumonia', 'No evidence of pneumonia',
                 'No evidence of acute pneumonia', 'No signs of pneumonia']

    return pos_query, neg_query


def save_img_from_url(image_url: str, local_path: Union[str, Path], md5: str = None) -> None:
    """
    Pull an image from a URL and save it to a local path
    """
    img_data = requests.get(image_url, timeout=30).content
    with open(local_path, 'wb') as handler:
        handler.write(img_data)

    if md5 is not None:
        assert check_integrity(local_path, md5)


def test_zero_shot_pneumonia_classification() -> None:
    """
    Checks latent similarity between text prompts and image embeddings for presence and absence of pneumonia
    """
    input_data = [
        ("https://openi.nlm.nih.gov/imgs/512/173/1777/CXR1777_IM-0509-1001.png", "f140126fff5d7d9f4a9402afadcbbf99", ClassType.PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/6/808/CXR808_IM-2341-2001.png", "ee19699d4305d17beecad94762a2ebcc", ClassType.PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/342/3951/CXR3951_IM-2019-1001.png", "786d5d854b1f6be1d6a0c3392794497a", ClassType.PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/16/2422/CXR2422_IM-0965-1001.png", "84dd31c1b0cbaaf8e1c0004d8917a78d", ClassType.PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/365/3172/CXR3172_IM-1494-1001.png", "dc476e73d3fd3b178a5303f48221e9d5", ClassType.NO_PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/161/1765/CXR1765_IM-0499-1001.png", "fdc6d3753f853352b35f5edf4ee0873c", ClassType.NO_PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/327/3936/CXR3936_IM-2007-1001.png", "ef537c81a0d8c0ae618625970c122ecc", ClassType.NO_PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/76/1279/CXR1279_IM-0185-1001.png", "9d03d740dcb0e7e068eb5eb73355262e", ClassType.NO_PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/5/5/CXR5_IM-2117-1003002.png", "204a2c83f94a4d4e74b6cea43caabdf2", ClassType.NO_PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/219/3828/CXR3828_IM-1932-1001.png", "1260a14f197527030fa0cb6b2d6950b8", ClassType.NO_PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/16/818/CXR818_IM-2349-1001.png", "2756b3a746e54e72f1efbbed72c2f83b", ClassType.PNEUMONIA),  # noqa: E501
        ("https://openi.nlm.nih.gov/imgs/512/358/2363/CXR2363_IM-0926-1001.png", "bcdc990161a5234e08d0595ed8a0bbf0", ClassType.PNEUMONIA)]   # noqa: E501

    img_txt_inference = _get_vlp_inference_engine()
    positive_prompts, negative_prompts = _get_default_text_prompts_for_pneumonia()

    for cxr_url, md5, label_str in input_data:
        suffix = Path(cxr_url).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            image_path = Path(f.name)
            save_img_from_url(cxr_url, image_path, md5=md5)

            positive_score = img_txt_inference.get_similarity_score_from_raw_data(
                image_path=image_path,
                query_text=positive_prompts)
            negative_score = img_txt_inference.get_similarity_score_from_raw_data(
                image_path=image_path,
                query_text=negative_prompts)

            if label_str == ClassType.PNEUMONIA:
                assert positive_score > negative_score
            else:
                assert negative_score > positive_score
