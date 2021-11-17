
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from datetime import datetime
from enum import Enum
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import jinja2
import ruamel.yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from health_azure.utils import (download_files_from_run_id, get_aml_run_from_run_id,
                                download_files_from_hyperdrive_children)

CLOSE_DOC_TAGS = "</p>\n</div>\n</body>\n</html>"
IMAGE_KEY_HTML = "IMAGEPATHSHTML"
TABLE_KEY_HTML = "TABLEKEYHTML"
REPORT_CONTENTS_KEY = "report_contents"


class ReportComponentKey(Enum):
    IMAGE = "image"
    IMAGE_GALLERY = "image_gallery"
    TABLE = "table"
    TEXT = "text"


class HTMLReport:
    """
    Create an HTML report which Azure ML is able to render. If you do not provide a title or output folder,
    this will generate a report at "outputs/report.html"
    """
    def __init__(self, title: str = "Report", output_folder: str = "outputs"):
        self.report_title = title
        self.output_folder = output_folder

        report_folder = Path(output_folder)  # / title.lower().replace(" ", "_")
        report_folder.mkdir(exist_ok=True, parents=True)

        self.report_folder = report_folder
        self.report_path_html = report_folder / (title.lower().replace(" ", "_") + '.html')
        self.report_html = ""
        self.template = ""
        self.template_path = self._create_template()

        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('/')
        )
        self.render_kwargs: Dict[str, Any] = {"title": title}

    def validate(self) -> None:
        """
        For our definition, the rendered HTML must contain exactly one open and closing tags doctype, head and body
        If any of these are missing from the repport_html, we raise a ValueError

        :return:
        """
        # If this function is called before the report is rendered, self.html_report will be empty.
        # calling render() will populate this attribute
        if len(self.report_html) == 0:
            self.render(save_html=False)
        expected_tags = ["<!DOCTYPE html>", "<head>", "</head>", "<body>", "</body>", "</html>"]
        for tag in expected_tags:
            if not self.report_html.count(tag) == 1:
                raise ValueError(f"report_html is missing the tag {tag}. This will cause problems with rendering")

    @staticmethod
    def _remove_html_end(report_stream: str) -> str:
        """
        Before adding additional components to the HTML report, we must remove the closing tags

        :param report_stream: A string representing the content of the report thus far
        :return: A modified string, without closing tags
        """
        return report_stream.replace(CLOSE_DOC_TAGS, "")

    def _create_template(self) -> Path:
        template_path = self.report_folder / "template.html"
        template_path.touch(exist_ok=True)

        self.template += """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
<title> {{title}} </title>
</head>
<body>
<div class="container-fluid justify-content-center align-items-center">
<div class="container">
<h1> {{title}} </h1>
</div>
<p>
""" + CLOSE_DOC_TAGS

        return template_path

    def add_text(self, text: str, tag_class: str = '') -> None:
        """
        Add text to the report content, in the form of a new paragraph. This will start on a new line by default.
        If you wish to provide your own css classes, you can tag this paragraph by providing the class name in the
        "tag_class" parameter.

        :param text: The text to add to the report
        :param tag_class: An optional class name to apply styling to the text
        """
        self.template = self._remove_html_end(self.template)
        p_tag_open = f"<p class={tag_class}>" if tag_class is not None else "<p>"
        self.template += f"""<div class="container" >
        {p_tag_open}{text}
        </div>
        <br>""" + CLOSE_DOC_TAGS

    def add_table(self, table: Optional[pd.DataFrame] = None, table_path: Optional[Path] = None) -> None:
        """
        Add a table object to your report. The table can either be passed as a Pandas DataFrame object, or
        a path to a .csv file contianing your table. If neither of these parameters are provided, an Exception
        will be raised.

        :param table: An optional Pandas DataFrame to be rendered in the report
        :param table_path: An optional path to a .csv file containing the table to be rendered on the report
        :raises ValueError: If neither a table object nor a path to a .csv file are provided
        """
        if table is None and table_path is None:
            raise ValueError("One of table or table path must be provided")
        if table is None:
            table = pd.read_csv(table_path)
        self.template = self._remove_html_end(self.template)

        num_existing_tables = self.template.count("table.to_html")
        table_key = f"{TABLE_KEY_HTML}_{num_existing_tables}"  # starts at zero

        self.template += """<div class="container" >
        {% for table in """ + table_key + """ %}
            {{ table.to_html(classes=[ "table"], justify="center") | safe }}
        {% endfor %}
        </div>
        <br>""" + CLOSE_DOC_TAGS

        self.render_kwargs.update({table_key: [table]})

    def add_image(self, image_path: str) -> None:
        """
        Given a path to an image file, embeds the image on the report. If the path is within the report folder, the
        relative path will be used. This is to ensure that the HTML document is able to locate and embed the image.
        Otherwise, the image path is not altered.

        :param image_path: The path to the image to be embedded
        """
        if image_path.startswith(str(self.report_folder)):
            img_path_html = Path(image_path).relative_to(self.report_folder)
        else:
            img_path_html = Path(image_path)

        self.template = self._remove_html_end(self.template)

        image_key_html = IMAGE_KEY_HTML + "_0"
        # Increment the image name so as not to replace other images
        num_existing_images = self.template.count("<img src=")

        image_key_html = image_key_html.split("_")[0] + f"_{num_existing_images}"

        self.template += """<div class="container">
        {% for image_path in """ + image_key_html + """ %}
            <img src={{image_path}} alt={{image_path}}>
        {% endfor %}
        </div>
        <br>""" + CLOSE_DOC_TAGS

        # Add these keys and paths to the keyword args for rendering later
        self.render_kwargs.update({image_key_html: [str(img_path_html)]})

    @classmethod
    def load_imgs_onto_subplot(cls, image_folder: Path, num_plot_columns: int = 2,
                               figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
        """
        Given a path to a folder containing multiple images, loads each of the images in the folder and
        adds to a single chart.

        :param image_folder: The folder containing the images to be plotted
        :param num_plot_columns: The number of columns of images to plot, defaults to 2
        :param figsize: The size of the overall figure , defaults to (12, 12)
        :return: A matplotlib Figure object
        """
        if num_plot_columns <= 1:
            raise ValueError("Can't have less than one column in your plot")

        plot_paths = list(image_folder.iterdir())

        num_plots = len(plot_paths)
        num_plot_rows = int(np.ceil(num_plots / num_plot_columns))
        fig = plt.figure(figsize=figsize)
        for i, j in itertools.product(range(num_plot_rows), range(num_plot_columns)):
            plot_index = (i * num_plot_columns) + j
            if plot_index >= num_plots:
                break
            plot_path = plot_paths[plot_index]
            with open(plot_path, "rb") as f_path:
                img_arr = plt.imread(f_path)

                fig.add_subplot(num_plot_rows, num_plot_columns, plot_index + 1)
                plt.imshow(img_arr)

        fig.tight_layout()
        return fig

    def add_image_gallery(self, image_folder: str) -> None:
        """
        For each image in a folder, create a "gallery" i.e. a plot containing all of these images,
        and add it to the report

        :param image_folder: The folder containing all of the images to add to the gallery
        """
        fig = self.load_imgs_onto_subplot(Path(image_folder), figsize=(15, 15))
        img_num = len(list(self.report_folder.glob("gallery_image_*")))
        gallery_img_path = str(self.report_folder / f"gallery_image_{img_num}.png")
        fig.savefig(gallery_img_path)
        self.add_image(gallery_img_path)

    def add_plot(self, plot_path: Optional[str] = None, fig: Optional[plt.Figure] = None) -> None:
        """
        Add a plot to your report. The plot can either be passed as a [matplotlib Figure object](
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html), or as a path to
        a saved plot file.

        :param plot_path: Optional path to a saved plot file
        :param fig: Optional matplotlib Figure object
        """
        if fig is not None:
            # save the plot
            if fig._suptitle is not None:
                plot_title = fig._suptitle.get_text().replace(" ", "_") + ".png"
            elif len(fig.texts) > 0:
                plot_title = fig.texts[0].get_text().replace(" ", "_") + ".png"
            else:
                plot_title = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            plot_path = self.report_folder / plot_title
            fig.tight_layout()
            fig.savefig(plot_path, bbox_inches='tight', dpi=150)

        self.add_image(str(plot_path))

    def read_config_yaml(self, report_config_path: Path) -> OrderedDict:
        """
        Load a report description from a yaml file and use it create an HTML report. The yaml file must as a minimum
        contain the section "report_contents", which takes a list of tuples of component types and their paths.
        For example, to add an image, the tuple might look like ("image", "path/to/image"). For a table, it might
        look like ("table", "path/to/csv/file"). The syntax for text is different, as the second component in
        the tuple is not a path, but the text itself. E.g. ("text", "A section heading"). Provided that one or more
        of these elements is found, self.template and self.render_kwargs will be updated

        :param report_config_path: The path to the .yml file contianing the report description
        :raises ValueError: If a tuple in the report_contents section as a first entry other than "image", "table"
            or "text"
        :return: An OrderedDict representing the contents of the yaml file
        """
        # TODO: add option to overwrite report title with entry here
        assert report_config_path.suffix == ".yml", f"Expected a .yml file but found {report_config_path.suffix}"
        with open(report_config_path, "r") as f_path:
            yaml_contents = ruamel.yaml.load(f_path)

        return yaml_contents

    def add_yaml_contents_to_report(self, yaml_contents: OrderedDict) -> None:
        """
        Given an OrderdDict of report contents containing the key "report_contents", where the value is
        a list of lists. Each one has 2 entries: firstly, a "content_type" - e.g. "image", "table" or "text".
        The second entry is a path to the content, in the case of "image" or "table" types, or a string in
        the case of text types. In the case of the paths, these can either be paths to a single image/csv file,
        or to a folder containing multiple images/ csv files. If multiple, they will be embedded successively.

        E.g.
        {"table_contents" :
            [
                ["image": "<path/to/image/file_or_folder>"],
                ["table", "<path/to/csv_file_or_folder>"],
                ["text", "A subsection header"]
            ]
        }

        The attribtues "template" and "render_kwargs" will be updated, as the respective methods add_image, add_table
        and add_text are called as necessary.

        :param yaml_contents: An OrderedDict containing at least a "report_contents" section
        :raises ValueError: If the first entry in a row of report_contents is something other than image, table or text
        """
        report_contents = yaml_contents[REPORT_CONTENTS_KEY]
        for component_type, component_val in report_contents:
            if component_type == ReportComponentKey.TABLE.value:
                table_path = Path(component_val)
                if table_path.is_dir():
                    for tbl_path in table_path.iterdir():
                        self.add_table(table_path=tbl_path)
                else:
                    self.add_table(table_path=table_path)
            elif component_type == ReportComponentKey.IMAGE.value:
                image_path = Path(component_val)
                if image_path.is_dir():
                    for img_path in image_path.iterdir():
                        self.add_image(str(img_path))
                else:
                    self.add_image(component_val)
            elif component_type == ReportComponentKey.IMAGE_GALLERY.value:
                self.add_image_gallery(component_val)
            elif component_type == ReportComponentKey.TEXT.value:
                self.add_text(component_val)
            else:
                raise ValueError("Key must either equal table, image or text")

    def download_report_contents_from_aml(self, run_id: str, report_contents: List[List[str]],
                                          hyperdrive_hyperparam_name: str = '') -> List[List[str]]:
        """
        Downloads report contents (images, csv files etc, as specified in the ) from Azure ML Runs. If the
        run_id provided represents an AML HyperDrive run, will attempt to download each of the specified paths
        in report_contents from each of the child runs. These will be saved into separate folders in the
        report folder. Note that to retrieve these runs, you must provide a value for `hyperdrive_hyperparam_name`
        - i.e. the name of a hyperparameter that was sampled over during this run.

        :param run_id: A string representing the run id of the AML run from which to download files
        :param report_contents: An List of Lists, where the first entry in each is the report content type
            (image, table, text) and the second is either a path to where the file lives in your DataStore,
            or else it is a string to be added to the report.
        :param hyperdrive_hyperparam_name: If the run is a hyperdrive run, specify the name of one of the
            hyperparameters that was sampled here. This is to ensure files are downloaded into logically-named
            folders.
        :return: An updated list of report contents, with paths replaced by the downloaded file paths where
            applicable
        """
        # If this is a hyperdrive run, download files for each of its children
        run = get_aml_run_from_run_id(run_id)

        # TODO: tuple -> list
        updated_report_contents = []
        for artifact_type, artifact_val in report_contents:
            # If the compoment is text, we don't need to download anything
            if artifact_type == ReportComponentKey.TEXT.value:
                updated_report_contents.append([artifact_type, artifact_val])
            else:
                if run.type == "hyperdrive":
                    full_artifact_path = download_files_from_hyperdrive_children(
                        run,
                        artifact_val,
                        self.report_folder,
                        hyperparam_name=hyperdrive_hyperparam_name,
                    )
                else:
                    full_artifact_path = self.report_folder / artifact_val
                    download_files_from_run_id(run_id, self.report_folder, prefix=artifact_val)

                updated_report_contents.append([artifact_type, str(full_artifact_path)])

        return updated_report_contents

    def render(self, save_html: bool = True) -> None:
        """
        Render the HTML report template with the keyword arguments that have been collected thorughout report
        generation. Unless save_html is False, creates a file at self.report_path_html and writes the HTML
        content to it

        :param save_html: Whether to save the HTML to file
        """
        subs: str = self.env.from_string(self.template).render(**self.render_kwargs)
        self.report_html = subs
        # write the substitution to a file
        if save_html:
            with open(self.report_path_html, 'w') as f_path:
                f_path.write(subs)
