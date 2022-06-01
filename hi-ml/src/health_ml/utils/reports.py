#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import base64
import itertools
import mimetypes
from datetime import datetime
from enum import Enum
from itertools import chain
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
DEFAULT_OUTPUTS_FOLDER = "outputs"
DEFAULT_FIGSIZE = (15, 15)
DEFAULT_NUM_COLS = 2


class ReportComponentKey(Enum):
    IMAGE = "image"
    IMAGE_GALLERY = "image_gallery"
    TABLE = "table"
    TEXT = "text"
    TYPE = "type"
    VALUE = "value"


class HTMLReport:
    """
    Create an HTML report which Azure ML is able to render. If you do not provide a title or output folder,
    this will generate a report at "outputs/report.html"
    """

    def __init__(self, title: str = "Report", output_folder: str = DEFAULT_OUTPUTS_FOLDER):
        self.report_title = title
        self.output_folder = output_folder

        report_folder = Path(output_folder)
        report_folder.mkdir(exist_ok=True, parents=True)

        self.report_folder = report_folder
        self.report_path_html = (report_folder / title.lower().replace(" ", "_")).with_suffix('.html')
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
            if self.report_html.count(tag) < 1:
                raise ValueError(f"report_html is missing the tag {tag}. This will cause problems with rendering")
            elif self.report_html.count(tag) > 1:
                raise ValueError(f"report_html contains more than one tag {tag}. This will cause problems with"
                                 "rendering")

    @staticmethod
    def _remove_html_end(report_stream: str) -> str:
        """
        Before adding additional components to the HTML report, we must remove the closing tags

        :param report_stream: A string representing the content of the report thus far
        :return: A modified string, without closing tags
        """
        return report_stream.replace(CLOSE_DOC_TAGS, "")

    def _add_html_end(self) -> None:
        """
        Once we have added components to our template, this method is called, to add back the HTML
        closing tags. validate is called to check the correct number of closing tags exist.

        :return: A modified string, with closing tags
        """
        self.template += f"{CLOSE_DOC_TAGS}"
        self.validate()

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
<div class="container-fluid">
<div class="container">
<h1> {{title}} </h1>
</div>
<p>
""" + CLOSE_DOC_TAGS

        return template_path

    def add_to_template(self, template_addition: str) -> None:
        """
        Performs the three necessary tasks to add a section to the template. Firstly, removes existing closing
        tags. Next, adds the content (passed as a string to this method). Finally, adds back the closing tags.

        :param template_addition: The string to be added to the template
        """
        self.template = self._remove_html_end(self.template)
        self.template += template_addition
        self._add_html_end()

    def add_heading(self, text: str, level: int = 2, tag_class: str = '') -> None:
        """
        Add a heading to the report content. If you wish to provide your own css classes, you can
        tag this heading by providing the class name in the "tag_class" parameter.

        :param text: The contents of the heading to add to the report
        :param level: The heading level, e.g. 2 for `<h2>` etc.
        :param tag_class: An optional class name to apply styling to the text
        """
        if level < 1 or level > 5:
            raise ValueError(f"Level must be an integer between 1 and 5 (inclusive), but got {level}")
        class_spec = f" class={tag_class}" if tag_class else ""
        template_addition = f"""<div class="container" >
        <h{level}{class_spec}>{text}</h{level}>
        </div>"""
        self.add_to_template(template_addition)

    def add_text(self, text: str, tag_class: str = '') -> None:
        """
        Add text to the report content, in the form of a new paragraph. This will start on a new line by default.
        If you wish to provide your own css classes, you can tag this paragraph by providing the class name in the
        "tag_class" parameter.

        :param text: The text to add to the report
        :param tag_class: An optional class name to apply styling to the text
        """
        p_tag_open = f"<p class={tag_class}>" if tag_class is not None else "<p>"

        template_addition = f"""<div class="container" >
        {p_tag_open}{text}
        </div>
        <br>"""
        self.add_to_template(template_addition)

    def _add_tables_to_report(self, tables: List[pd.DataFrame]) -> None:
        """
        Add one or more tables (in the form of Pandas DataFrames) to the report.

        :param tables: A list of one or more Pandas DataFrame to be rendered in the report
        """
        for table in tables:
            num_existing_tables = self.template.count("table.to_html")
            table_key = f"{TABLE_KEY_HTML}_{num_existing_tables}"  # starts at zero

            template_addition = """<div class="container" >
            {% for table in """ + table_key + """ %}
                {{ table.to_html(classes=[ "table"], justify="center") | safe }}
            {% endfor %}
            </div>
            <br>"""
            self.add_to_template(template_addition)

            self.render_kwargs.update({table_key: [table]})

    def add_tables(self, tables: Optional[List[pd.DataFrame]] = None,
                   table_paths_or_dir: Optional[List[Path]] = None) -> None:
        """
        Add one or more tables to your report. The table can either be passed as a Pandas DataFrame object, or
        a list of path to one or more .csv files, or a directory of csv files containing your tables.
        If neither of these parameters are provided, an Exception will be raised.

        :param tables: An optional list of one or more Pandas DataFrames to be rendered in the report
        :param table_paths_or_dir: An optional list of one or more paths to .csv files containing the tables
            to be rendered on the report
        :raises ValueError: If neither a list of tables nor a list of paths is provided
        """
        if tables is None and table_paths_or_dir is None:
            raise ValueError("One of tables or table_paths_or_dir must be provided")

        tables = tables or []
        if table_paths_or_dir is not None:
            for table_path_or_dir in table_paths_or_dir:
                table_path_or_dir = Path(table_path_or_dir)
                if table_path_or_dir.is_dir():
                    for table_path in table_path_or_dir.iterdir():
                        self.add_tables(table_paths_or_dir=[table_path])
                    return
                else:
                    tables.append(pd.read_csv(table_path_or_dir))

        assert len(tables) > 0, "No tables were found"
        self._add_tables_to_report(tables)

    def _add_image_to_report(self, image_path: Path, base64_encode: bool = False) -> None:
        """
        Given a path to an image, add it to the report template and to the report arguments
        for rendering later. Optionally encode as base64 - this is useful if a standalone
        report is required but leads to a much larger report file size, so is False by default

        :param image_path: The paths to the image to be added
        :param base64_encode: If True, encode the image as base64 in the HTML report. Default is False
        """
        if self.report_folder in image_path.parents:
            img_path_html = image_path.relative_to(self.report_folder)
        else:
            img_path_html = image_path

        image_key_html = IMAGE_KEY_HTML + "_0"
        # Increment the image name so as not to replace other images
        num_existing_images = self.template.count("<img src=")

        image_key_html = image_key_html.split("_")[0] + f"_{num_existing_images}"

        template_addition = """<div class="container">
        {% for image_path in """ + image_key_html + """ %}
            <img src={{image_path}} alt={{image_path}}>
        {% endfor %}
        </div>
        <br>"""
        self.add_to_template(template_addition)

        # Add these keys and paths to the keyword args for rendering later
        if base64_encode:
            with open(image_path, "rb") as f_path:
                img_data = f_path.read()
                img_data_base64_bytes = base64.b64encode(img_data)
                img_data_base64_str = img_data_base64_bytes.decode()

            img_type: str = mimetypes.guess_type(str(img_path_html))[0]  # type: ignore
            img_path_str = "data:" + img_type + ";base64," + img_data_base64_str
        else:
            img_path_str = str(img_path_html)
        self.render_kwargs.update({image_key_html: [img_path_str]})

    def add_images(self, image_paths_or_dir: List[Path], base64_encode: bool = False) -> None:
        """
        Given a path to one or more image files, or a directory containing image files, embeds the image on the
        report. If the path is within the report folder, the relative path will be used.
        This is to ensure that the HTML document is able to locate and embed the image.
        Otherwise, the image path is not altered.

        :param image_paths_or_dir: The paths to the image(s), or a directory containing images to be embedded
        :param base64_encode: If True, encode image data as base64 in the HTML report. Default is False
        """
        if len(image_paths_or_dir) == 0:
            raise ValueError("add_image expects a list of image_paths")

        for image_path_or_dir in image_paths_or_dir:
            image_path_or_dir = Path(image_path_or_dir)
            if image_path_or_dir.is_dir():
                for image_path in image_path_or_dir.iterdir():
                    self.add_images([image_path], base64_encode=base64_encode)
            else:
                self._add_image_to_report(image_path_or_dir, base64_encode=base64_encode)

    @classmethod
    def load_imgs_onto_subplot(cls, img_folder_or_paths: List[Path], num_plot_columns: int = 2,
                               figsize: Tuple[int, int] = DEFAULT_FIGSIZE) -> plt.Figure:
        """
        Given a list of one or more paths, either to a folder containing multiple images, or multiple image
        paths, loads each of the images adds to a single chart

        :param img_folder_or_paths: A list containing either a single path to a folder containing all of the images to
            add to the gallery, or multiple paths to specific image files
        :param num_plot_columns: The number of columns of images to plot, defaults to 2
        :param figsize: The size of the overall figure
        :return: A matplotlib Figure object
        """
        if num_plot_columns <= 1:
            raise ValueError("Can't have less than one column in your plot")

        # if a single path is provided, we take its contents as the plot paths. If a list of image paths is provided
        # we use that.
        plot_paths: List[Path] = []
        for folder_or_path in img_folder_or_paths:
            if folder_or_path.is_dir():
                # Note: this will add every file within the named folder. Make sure these are only image files.
                plot_paths.extend(list(chain(list(folder_or_path.rglob("*")))))
            else:
                plot_paths.append(folder_or_path)

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

                axs: plt.Axes = fig.add_subplot(num_plot_rows, num_plot_columns, plot_index + 1)
                axs.set_axis_off()
                plt.imshow(img_arr)

        plt.tight_layout()
        return fig

    def add_image_gallery(self, image_folder_or_paths: List[Path], figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
                          num_cols: int = DEFAULT_NUM_COLS, base64_encode: bool = False) -> None:
        """
        Given a list of one or more paths, either to a folder containing multiple images, or multiple image
        paths, loads each of the images adds to a single chart create a "gallery" i.e. a plot containing
        all of these images, and add it to the report

        :param image_folder_or_paths: A list containing either a single path to a folder containing all of the images
            to add to the gallery, or multiple paths to specific image files
        :param figsize: The size of the overall plot
        :param num_cols: The number of columns in the subplot
        :param base64_encode: If True, encode image as base64 in the report HTML. Default is False.
        """
        fig = self.load_imgs_onto_subplot(image_folder_or_paths, figsize=figsize, num_plot_columns=num_cols)
        img_num = len(list(self.report_folder.glob("gallery_image_*")))
        gallery_img_path = self.report_folder / f"gallery_image_{img_num}.png"
        fig.savefig(str(gallery_img_path))
        self.add_images([gallery_img_path], base64_encode=base64_encode)

    def add_plot(self, plot_path: Optional[Path] = None, fig: Optional[plt.Figure] = None,
                 fig_title: Optional[str] = None) -> None:
        """
        Add a plot to your report. The plot can either be passed as a [matplotlib Figure object](
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html), or as a path to
        a saved plot file.

        :param plot_path: Optional path to a saved plot file
        :param fig: Optional matplotlib Figure object
        :param fig_title: Optionally provide a title to use as the saved figure filename
        """
        if fig is not None:
            # save the plot
            if fig._suptitle is not None:
                plot_title = fig._suptitle.get_text().replace(" ", "_")
            elif len(fig.texts) > 0:
                plot_title = fig.texts[0].get_text().replace(" ", "_")
            elif fig_title is not None:
                plot_title = fig_title.replace(" ", "_")
            else:
                plot_title = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            plot_path = (self.report_folder / plot_title).with_suffix(".png")
            fig.tight_layout()
            fig.savefig(str(plot_path), bbox_inches='tight', dpi=150)
        assert plot_path is not None  # for pyright
        self.add_images([plot_path])

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
        Given an OrderdDict of report contents, add these to the report. report_contents is an OrderedDict
        object containing the key "report_contents", whose value is a list of dictionaries. Each dictionary
        has at least 2 entries: firstly, a "type" - e.g. "image", "image_gallery", "table" or "text". The
        second entry each dictionary has is "value". In the case of "text" types, this is the
        text to render. In the case of "image", "image_gallery" or "table" types, this is a path to the content.
        These can either be paths to a single image/csv file, or to a folder containing multiple images/ csv
        files. If multiple, they will be embedded successively, unless the type "image_gallery" has been
        specified

        Some entries may have additional entries, such as "num_columns" or "figsize" for an image_gallery.

        E.g.
        {"table_contents" :
            [
                {"type": "image", "value": "<path/to/image/file_or_folder>"},
                {"type": "table", "value": "<path/to/csv_file_or_folder>"},
                {"type": "text", "value": "A subsection header"}
            ]
        }

        The attribtues "template" and "render_kwargs" will be updated, as the respective methods add_image, add_table
        and add_text are called as necessary.

        :param yaml_contents: An OrderedDict containing at least a "report_contents" section
        :raises ValueError: If the first entry in a row of report_contents is something other than image, table or text
        """
        report_contents = yaml_contents[REPORT_CONTENTS_KEY]
        for component in report_contents:
            component_type = component[ReportComponentKey.TYPE.value]
            component_val = component[ReportComponentKey.VALUE.value]
            base64_encode = component["base64_encode"] if "base64_encode" in component else False
            figsize = component["figsize"] if "figsize" in component else DEFAULT_FIGSIZE
            num_cols = component["num_cols"] if "num_cols" in component else DEFAULT_NUM_COLS
            dir_or_paths = [Path(x) for x in str(component_val).split(",")]
            if component_type == ReportComponentKey.TABLE.value:
                self.add_tables(table_paths_or_dir=dir_or_paths)
            elif component_type == ReportComponentKey.IMAGE.value:
                self.add_images(dir_or_paths, base64_encode=base64_encode)
            elif component_type == ReportComponentKey.IMAGE_GALLERY.value:
                self.add_image_gallery(dir_or_paths, figsize=figsize, num_cols=num_cols, base64_encode=base64_encode)
            elif component_type == ReportComponentKey.TEXT.value:
                self.add_text(component_val)
            else:
                raise ValueError("Key must either equal table, image or text")

    def download_report_contents_from_aml(self, run_id: str, report_contents: List[Dict[str, Any]],
                                          hyperdrive_hyperparam_name: str = '') -> List[Dict[str, Any]]:
        """
        Downloads report contents (images, csv files etc, as specified in the ) from Azure ML Runs. If the
        run_id provided represents an AML HyperDrive run, will attempt to download each of the specified paths
        in report_contents from each of the child runs. These will be saved into separate folders in the
        report folder, and the return value will include a comma separated string of the paths to each
        downloaded file. Note that to retrieve these runs, you must provide a value for `hyperdrive_hyperparam_name`
        - i.e. the name of a hyperparameter that was sampled over during this run.

        If not a HyperDrive run, a single file or folder will be downloaded,


        :param run_id: A string representing the run id of the AML run from which to download files
        :param report_contents: A list of dictionaries, where the "type" entry in each is the report content type
            (image, table, text) and the "value" entry is either a path to where the file lives in your DataStore,
            or else it is a string to be added to the report.
        :param hyperdrive_hyperparam_name: If the run is a hyperdrive run, specify the name of one of the
            hyperparameters that was sampled here. This is to ensure files are downloaded into logically-named
            folders.
        :return: An updated list of report contents, with paths replaced by the downloaded file paths where
            applicable.
        """
        # If this is a hyperdrive run, download files for each of its children
        run = get_aml_run_from_run_id(run_id)

        updated_report_contents = []
        for component in report_contents:
            component_type = component[ReportComponentKey.TYPE.value]
            component_val = component[ReportComponentKey.VALUE.value]
            # If the component is text, we don't need to download anything
            if component_type == ReportComponentKey.TEXT.value:
                updated_report_contents.append({ReportComponentKey.TYPE.value: component_type,
                                                ReportComponentKey.VALUE.value: component_val})
            else:
                if run.type == "hyperdrive":
                    artifact_paths = download_files_from_hyperdrive_children(
                        run,
                        component_val,
                        self.report_folder,
                        hyperparam_name=hyperdrive_hyperparam_name,
                    )
                    full_artifact_path = ",".join(artifact_paths)
                else:
                    full_artifact_path = str(self.report_folder / component_val)
                    download_files_from_run_id(run_id, self.report_folder, prefix=component_val)

                updated_component = {ReportComponentKey.TYPE.value: component_type,
                                     ReportComponentKey.VALUE.value: full_artifact_path}

                # add back any other entries such as figsize, num_columns etc
                additional_keys = set(component.keys()).difference(set(updated_component.keys()))
                for k in additional_keys:
                    updated_component[k] = component[k]
                updated_report_contents.append(updated_component)

        return updated_report_contents

    def render(self, save_html: bool = True) -> None:
        """
        Render the HTML report template with the keyword arguments that have been collected thorughout report
        generation. Unless save_html is False, creates a file at self.report_path_html and writes the HTML
        content to it

        :param save_html: Whether to save the HTML to file
        """
        new_env = self.env.from_string(self.template)
        subs: str = new_env.render(**self.render_kwargs)
        self.report_html = subs
        # write the substitution to a file
        if save_html:
            with open(self.report_path_html, 'w') as f_path:
                f_path.write(subs)

    def zip_report_folder(self) -> Path:
        """
        Zip the report folder at the location '<report_folder>.zip', preserving the directory structure.
        Returns the path to the zipped folder

        :return: The path to the zipped folder
        """
        import zipfile
        report_files = self.report_folder.rglob("*.*")
        zipped_folder_path = self.report_folder.with_suffix(".zip")
        with zipfile.ZipFile(zipped_folder_path, "w") as zipped_folder:
            for report_file in report_files:
                zipped_folder.write(report_file, arcname=report_file.relative_to(self.report_folder))
        print(f"Zipped folder path: {str(zipped_folder_path)}")
        return zipped_folder_path
