#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Tools to upload training results from Azure Machine Learning runs to a deployed instance of the
`Digital Slide Archive <https://digitalslidearchive.github.io/>`_.

The `Girder Python client <https://girder.readthedocs.io/en/latest/python-client.html>`_ is used
to communicate with the API.

An example API can be found at the `Kitware demo <https://demo.kitware.com/histomicstk/api/v1>`_.
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import tempfile
import webbrowser
from pathlib import Path
from urllib.parse import urljoin
from dataclasses import dataclass, astuple, asdict
from typing import Any, Dict, List, Optional, Sequence, Union

import azureml
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from girder_client import GirderClient
from health_azure.logging import logging_to_stdout
from health_azure.utils import get_aml_run_from_run_id

from health_cpath.utils.naming import ResultsKey
from health_cpath.utils.output_utils import AML_TEST_OUTPUTS_CSV


TypeRectangleJSON = Dict[str, Union[str, float, Dict[str, str]]]
TypePointJSON = Dict[str, Union[str, List[float], Dict[str, str]]]
TypeAnnotationJSON = Dict[str, Union[str, List[Dict]]]

# DSA coordinates are generally expressed using three dimensions
Z_ZERO = [0]


@dataclass
class Color:
    """Container for RGBA color.

    All values are expected to be in :math:`[0, 255]`. Check the
    `documentation <https://github.com/girder/large_image/blob/master/girder_annotation/docs/annotations.rst#colors>`_
    for more information.
    """
    red: int
    green: int
    blue: int
    alpha: int

    def __post_init__(self) -> None:
        array = np.asarray(self)
        if (array < 0).any() or (array > 255).any():
            raise ValueError(f"All RGBA components must be between 0 and 255, but {astuple(self)} were passed")

    def __array__(self) -> np.ndarray:
        return np.array(self.components, dtype=int)

    @property
    def components(self) -> Sequence[int]:
        """Tuple of R, G, B and A components."""
        return astuple(self)

    def __str__(self) -> str:
        return f"rgba{astuple(self)}"


TRANSPARENT = Color(0, 0, 0, 0)


@dataclass
class Element:
    """Base class for annotations elements such as points or rectangles."""
    label: str
    fill_color: Color
    line_color: Color

    def as_json(self) -> Dict:
        """Return JSON representation suitable for the DSA."""
        raise NotImplementedError


@dataclass
class Coordinates:
    """Helper class to represent x and y coordinates."""
    x: float
    y: float

    def __post_init__(self) -> None:
        try:
            float(self.x)
            float(self.y)
        except ValueError as e:
            raise TypeError(f"Error converting coordinates to float: \"{asdict(self)}\"") from e

    def __array__(self) -> np.ndarray:
        return np.asarray(astuple(self))


@dataclass
class Rectangle(Element):
    """Container for rectangles in annotations.

    More information can be found in the
    `DSA documentation <https://github.com/girder/large_image/blob/master/girder_annotation/docs/annotations.rst#rectangle>`_.
    """  # noqa: E501
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.left >= self.right:
            raise ValueError(f"The value for right ({self.right}) must be larger than the value for left ({self.left})")
        if self.top >= self.bottom:
            raise ValueError(f"The value for bottom ({self.bottom}) must be larger than the value for top ({self.top})")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_xy(self) -> np.ndarray:
        size_xy = np.asarray((self.width, self.height))
        left_top = self.left, self.top
        return np.asarray(left_top) + size_xy / 2

    def as_json(self) -> TypeRectangleJSON:
        data: TypeRectangleJSON = {}
        data["fillColor"] = str(self.fill_color)
        data["lineColor"] = str(self.line_color)
        data["type"] = "rectangle"
        data["center"] = self.center_xy.tolist() + Z_ZERO
        data["width"] = float(self.width)
        data["height"] = float(self.height)
        data["label"] = {"value": self.label}
        return data


@dataclass
class Point(Element):
    """Container for points in DSA annotations.

    More information can be found in the
    `DSA documentation <https://github.com/girder/large_image/blob/master/girder_annotation/docs/annotations.rst#point>`_.
    """  # noqa: E501
    center: Coordinates

    def __post_init__(self) -> None:
        if not isinstance(self.center, Coordinates):
            raise TypeError(f"Center must be an instance of Coordinates, not {type(self.center)}")

    def as_json(self) -> TypePointJSON:
        data: TypePointJSON = {}
        data["fillColor"] = str(self.fill_color)
        data["lineColor"] = str(self.line_color)
        data["type"] = "point"
        data["center"] = list(astuple(self.center)) + Z_ZERO
        data["label"] = {"value": self.label}
        return data


@dataclass
class Annotation:
    """Container for DSA annotation.

    An example can be found in the
    `DSA documentation <https://github.com/girder/large_image/blob/master/girder_annotation/docs/annotations.rst#a-sample-annotation>`_.
    """  # noqa: E501
    name: str
    elements: Sequence[Element]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            # This is enforced by the JSON schema
            raise ValueError("The annotation name cannot be empty")

    def as_json(self) -> TypeAnnotationJSON:
        data: TypeAnnotationJSON = {}
        data["name"] = self.name
        data["description"] = self.description
        data["elements"] = [element.as_json() for element in self.elements]  # type: ignore
        return data


class DigitalSlideArchive:
    """Representation of a deployed instance of the Digital Slide Archive.

    :param url: URL of a deployed instance of the `Digital Slide Archive <https://digitalslidearchive.github.io/>`_.
        For example: https://demo.kitware.com/histomicstk/. If not given, it must be defined in an environment
        variable ``DSA_URL``.
    :param api_key: Girder `API key <https://girder.readthedocs.io/en/latest/user-guide.html#api-keys>`_ to
        perform allowed operations. If not given, it must be defined in an environment
        variable ``DSA_API_KEY``.
    """
    URL_ENV_NAME = "DSA_URL"
    API_KEY_ENV_NAME = "DSA_API_KEY"

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            url = os.environ[self.URL_ENV_NAME] if url is None else url
        except KeyError as e:
            message = (
                "The DSA URL must be passed as an argument"
                f" or stored in an environment variable \"{self.URL_ENV_NAME}\""
            )
            raise RuntimeError(message) from e
        self.url = url
        self._client = self._get_client(self.api_url, api_key)
        self.post = self._client.post
        self.get = self._client.get

    def _get_client(self, api_url: str, api_key: Optional[str]) -> GirderClient:
        logging.info("Logging into DSA API hosted at %s...", api_url)
        client = GirderClient(apiUrl=api_url)
        try:
            api_key = os.environ[self.API_KEY_ENV_NAME] if api_key is None else api_key
        except KeyError as e:
            message = (
                "The DSA API key must be passed as an argument"
                f" or stored in an environment variable \"{self.API_KEY_ENV_NAME}\""
            )
            raise RuntimeError(message) from e
        client.authenticate(apiKey=api_key)
        logging.info("Authentication successful")
        return client

    @property
    def api_url(self) -> str:
        return urljoin(self.url, GirderClient.DEFAULT_API_ROOT)

    def add_annotation(self, item_id: str, annotation: Annotation) -> Dict:
        """Add annotation to a DSA item.

        :param item_id: Unique string representing the item ID.
        :param annotation: Instance of :class:`Annotation` that will be added to the item.
        """
        response = self.post(
            path="annotation",
            parameters={"itemId": item_id},
            json=annotation.as_json(),
        )
        return response

    def search_item(self, text: str, search_mode: str = "text") -> Item:
        """Search a file in the DSA collections and return its parent item.

        :param text: Text query.
        :param search_mode: Girder search mode. It ``'text'``, the full item ID will be matched.
            If ``'prefix'``, the file whose name starts with the value in ``text`` will be returned.
        :raises RuntimeError: If no items are found for the query or if more than one item is found.
        """
        parameters = dict(q=text, types="[\"item\"]", mode=search_mode)
        result = self.get("/resource/search", parameters=parameters)
        items_jsons = result["item"]
        if not items_jsons:
            raise RuntimeError(f"No items found for query \"{text}\"")
        elif len(items_jsons) > 1:
            raise RuntimeError(f"More than one item found for query \"{text}\":\n{items_jsons}")
        return Item(self, json=items_jsons[0])


class Item:
    """Representation of an item in the Digital Slide Archive.

    :param dsa: Instance of :class:`DigitalSlideArchive`.
    :param id: ID of the item in the Digital Slide Archive.
    :param json: JSON representation of the Digital Slide Archive item.
    :raises ValueError: If no ID or JSON is passed.
    :raises ValueError: If both an ID and JSON are passed.
    """

    def __init__(self, dsa: DigitalSlideArchive, id: Optional[str] = None, json: Optional[Dict] = None):
        self._dsa = dsa

        # Validate that only one of ID or the JSON dict have been passed
        if id is not None and json is not None:
            raise ValueError("Only the ID or the JSON object can be passed")
        if id is None and json is None:
            raise ValueError("An ID or JSON object must be passed")

        if id is None:
            id = json["_id"]  # type: ignore
        self.id = id
        self._json = json

    @property
    def url(self) -> str:
        return urljoin(self._dsa.url, f"/#item/{self.id}")

    def open(self) -> bool:
        return webbrowser.open(self.url)

    def add_annotation(self, annotation: Annotation) -> Dict:
        response = self._dsa.post(
            path="annotation",
            parameters={"itemId": self.id},
            json=annotation.as_json(),
        )
        return response


class RunOutputs:
    """Class to process outputs CSV of an Azure Machine Learning (AML) run.

    :param run_id: ID of the AML run from which results will be downloaded.
    :param workspace_config_path: Path to an AML workspace configuration file, e.g., ``config.json``.
    :param overwrite_csv: Force download of the output CSV even when it is found locally.
    """

    def __init__(
        self,
        run_id: str,
        workspace_config_path: Optional[Path] = None,
        overwrite_csv: bool = False,
    ):
        logging.info("Getting run \"%s\"...", run_id)
        run = get_aml_run_from_run_id(run_id, workspace_config_path=workspace_config_path)
        experiment = run.experiment
        workspace = experiment.workspace

        self.run = run
        self.experiment = experiment
        self.workspace = workspace
        self.df = self.get_df(overwrite_csv)
        self.tile_size = None

    def get_df(self, overwrite_csv: bool) -> pd.DataFrame:
        """Download outputs CSV from Azure ML and read the data frame.

        The CSV is cached locally for future

        :param overwrite_csv: Force download of the output CSV even when it is found locally.
        """
        csv_filename = AML_TEST_OUTPUTS_CSV
        csv_stem = Path(csv_filename).stem
        csv_name = f"{csv_stem}-{self.workspace.name}-{self.run.id}.csv"
        cached_csv_path = Path(tempfile.gettempdir()) / csv_name
        if cached_csv_path.is_file() and not overwrite_csv:
            logging.info("Found cached CSV file")
        else:
            logging.info("Downloading outputs CSV...")
            aml_exceptions = (
                azureml.exceptions._azureml_exception.UserErrorException,
                azureml._restclient.models.error_response.ErrorResponseException,
            )
            try:
                self.run.download_file(csv_filename, cached_csv_path)
            except aml_exceptions as e:
                raise FileNotFoundError("Error downloading outputs file from run") from e
        logging.info("Reading CSV file: %s ...", cached_csv_path)
        return self._read_csv(cached_csv_path)

    def get_annotation_from_slide_data_frame(
        self,
        df: pd.DataFrame,
        name: str,
        rescale: bool = False,
        colormap_name: str = "Greens",
        description: str = "",
    ) -> Annotation:
        """Create an annotation from a slide data frame.

        :param df: Data frame with data from a single slide.
        :param name: Annotation name.
        :param rescale: If ``True``, attention values will be remapped to :math:`[0, 1]` to increase
            the dynamic range of output colors.
        :param colormap_name: Name of the Matplotlib colormap used for the annotation elements.
        :param description: Optional description for the annotation.
        """
        original_attentions = df.bag_attn.values
        if rescale:
            df = df.copy()
            attentions = df.bag_attn.values
            df.bag_attn = (attentions - attentions.min()) / np.ptp(attentions)
        colormap = plt.get_cmap(colormap_name)
        rectangles = []
        for i, (_, row) in enumerate(df.iterrows()):
            rgba_uchar = colormap(row.bag_attn, bytes=True)
            fill_color = Color(*rgba_uchar)
            line_color = TRANSPARENT
            rectangle = Rectangle(
                f"{original_attentions[i]:.4f}",
                fill_color,
                line_color,
                left=row.left,
                top=row.top,
                right=row.right,
                bottom=row.bottom,
            )
            rectangles.append(rectangle)
        return Annotation(name, rectangles, description=description)

    @staticmethod
    def _read_csv(csv_path: Path) -> pd.DataFrame:
        return pd.read_csv(csv_path, index_col=0)

    def get_slide_annotation_from_df(
        self,
        df_or_path: Union[pd.DataFrame, Path],
        slide_id: str,
        annotation_name: str,
        **annotation_kwargs: Any,
    ) -> Annotation:
        """Create an annotation from an outputs data frame.

        :param df_or_path: Data frame or path to a CSV file.
        :param slide_id: Slide ID as described in the data frame.
        :param annotation_name: Name of the generated annotation.
        :param annotation_kwargs: Additional kwargs to :meth:`get_annotation_from_slide_data_frame`.
        """
        df = df_or_path if isinstance(df_or_path, pd.DataFrame) else self._read_csv(df_or_path)
        slide_mask = df[ResultsKey.SLIDE_ID] == slide_id
        df_slide = df[slide_mask]
        return self.get_annotation_from_slide_data_frame(df_slide, annotation_name, **annotation_kwargs)

    def upload(
        self,
        dsa: DigitalSlideArchive,
        dry_run: bool = False,
        max_slides: Optional[int] = None,
        id_filter: Optional[str] = "",
        search_mode: str = "full",
        **annotation_kwargs: Any,
    ) -> List[Dict]:
        """Create annotations from a data frame and upload them to DSA.

        :param dsa: Instance of :class:`DigitalSlideArchive` to which results will be uploaded.
        :param dry_run: Create annotations and show outputs, without uploading any data.
        :param max_slides: Maximum number of slides to upload, useful for debugging.
        :param id_filter: Filter to only process slides matching this string, according to ``search_mode``.
        :param search_mode: See :meth:`DigitalSlideArchive.search_item`.
        :param annotation_kwargs: Additional kwargs to :meth:`get_annotation_from_slide_data_frame`.
        """
        unique_slide_ids = sorted(self.df[ResultsKey.SLIDE_ID].unique())

        num_slides = len(unique_slide_ids)
        logging.info("Found outputs for %s slides", num_slides)
        if max_slides is not None:
            max_slides = min(max_slides, num_slides)
            percentage = 100 * max_slides / num_slides
            logging.info("Using %s/%s slides (%s %% of total)", max_slides, num_slides, f"{percentage:.1f}")
            unique_slide_ids = unique_slide_ids[:max_slides]

        # I think "full" is more descriptive than "text" for our API
        search_mode = "text" if search_mode == "full" else search_mode
        progress = tqdm(unique_slide_ids)
        responses = []
        for slide_id in progress:
            progress.set_description(slide_id)
            if id_filter not in slide_id:
                continue
            item = dsa.search_item(slide_id, search_mode=search_mode)
            tqdm.write(f"Processing slide {slide_id} - {item.url}")
            annotation_name = self.run.id
            annotation = self.get_slide_annotation_from_df(
                self.df,
                slide_id,
                annotation_name,
                **annotation_kwargs,
            )
            if not dry_run:
                responses.append(item.add_annotation(annotation))
        return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dsa_class = DigitalSlideArchive
    parser.add_argument(
        "--run-id",
        type=str,
        help="Azure ML run ID",
    )
    parser.add_argument(
        "--dsa-url",
        type=str,
        help=f"URL to a Digital Slide Archive deployment. If not passed, it must be set in {dsa_class.URL_ENV_NAME}",
    )
    parser.add_argument(
        "--dsa-key",
        type=str,
        help=f"API key with edit permissions. If not passed, it must be set in {dsa_class.API_KEY_ENV_NAME}",
    )
    parser.add_argument(
        "--workspace-config",
        type=Path,
        help="Path to workspace configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--overwrite_csv",
        action="store_true",
        help="Force the download of the outputs CSV even when it already exists in the cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Iterate over slides without uploading annotations",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        help="Maximum number of slides to process (useful for development)",
    )
    parser.add_argument(
        "--id-filter",
        type=str,
        help="Only process slides whose ID contain this substring",
        default="",
    )
    parser.add_argument(
        "--login-only",
        action="store_true",
        help="Just log into the DSA and exit. Useful to ensure connection to the DSA from current host",
    )
    parser.add_argument(
        "--rescale",
        action="store_true",
        help="Rescale attention values between 0 and 1 to maximize heatmaps contrast",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        choices=plt.colormaps(),
        default="Greens",
        help="Matplotlib colormap used for the heatmaps",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        choices=("full", "prefix"),
        default="full",
        help=(
            "If \"full\", the slide ID must match the DSA file name with or without extension."
            " If \"prefix\", all files whose name starts with the slide ID will be matched"
        )
    )
    args = parser.parse_args()

    logging_to_stdout()
    dsa = DigitalSlideArchive(args.dsa_url, args.dsa_key)
    if args.login_only:
        sys.exit(0)
    if args.run_id is None:
        raise ValueError("Please specify an Azure Machine Learning run ID")
    outputs = RunOutputs(
        run_id=args.run_id,
        workspace_config_path=args.workspace_config,
        overwrite_csv=args.overwrite_csv,
    )
    outputs.upload(
        dsa,
        dry_run=args.dry_run,
        max_slides=args.max_slides,
        id_filter=args.id_filter,
        search_mode=args.search_mode,
        colormap_name=args.colormap,
        rescale=args.rescale,
    )
