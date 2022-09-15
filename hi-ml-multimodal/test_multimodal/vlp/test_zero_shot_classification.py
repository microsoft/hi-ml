import tempfile
from pathlib import Path
from typing import Tuple, List

from torchvision.datasets.utils import download_url
from health_multimodal.image import ImageInferenceEngine, ImageModel, ResnetType
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.model.model import get_biovil_resnet
from health_multimodal.text.utils import get_cxr_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine

RESIZE = 512
CENTER_CROP_SIZE = 480


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


def test_zero_shot_pneumonia_classification() -> None:
    """
    Checks latent similarity between text prompts and image embeddings for presence and absence of pneumonia
    """
    cxr_url = "https://prod-images-static.radiopaedia.org/images/1371188/0a1f5edc85aa58d5780928cb39b08659c1fc4d6d7c7dce2f8db1d63c7c737234_gallery.jpeg"  # noqa: E501

    with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
        image_path = Path(f.name)
        download_url(
            cxr_url,
            str(image_path.parent),
            image_path.name,
            md5="5ba78c33818af39928939a7817541933",
        )
        img_txt_inference = _get_vlp_inference_engine()
        positive_prompts, negative_prompts = _get_default_text_prompts_for_pneumonia()

        positive_score = img_txt_inference.get_similarity_score_from_raw_data(
            image_path=image_path,
            query_text=positive_prompts,
        )
        negative_score = img_txt_inference.get_similarity_score_from_raw_data(
            image_path=image_path,
            query_text=negative_prompts,
        )

        assert positive_score > negative_score
