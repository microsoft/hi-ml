import json
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any, Iterable, Type, Callable

import jinja2
import numpy as np
import pandas as pd
# import pypandoc
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve
from xhtml2pdf import pisa

from fpdf import FPDF

IMAGE_KEY_HTML = "image_paths_html"
IMAGE_KEY_PDF = "image_paths_pdf"
VALID_PNG_EXTENSIONS = [".png"]
VALID_NUMPY_EXTENSIONS = (".npy", ".npz")

def _file_matches_extension(file: Path, valid_extensions: Iterable[str]) -> bool:
    """
    Returns true if the given file name has any of the provided file extensions.

    :param file: The file name to check.
    :param valid_extensions: A tuple with all the extensions that are considered valid.
    :return: True if the file has any of the given extensions.
    """
    dot = "."
    extensions_with_dot = tuple(e if e.startswith(dot) else dot + e for e in valid_extensions)
    return str(file).lower().endswith(extensions_with_dot)


def is_numpy_file_path(file: Path) -> bool:
    """
    Returns true if the given file name appears to belong to a Numpy file.

    :param file: The file name to check.
    :return: True if the file name indicates a Numpy file.
    """
    return _file_matches_extension(file, VALID_NUMPY_EXTENSIONS)


def is_png(file: Path) -> bool:
    """
    Returns true if file is png
    """
    return _file_matches_extension(file, VALID_PNG_EXTENSIONS)


def load_png(path: Path, mode='RGB') -> Image:
    im = Image.open(path)
    return im.convert(mode)


def load_numpy_image(path: Path, image_type: Optional[Type] = None) -> Image:
    image = np.load(path)
    if type(image) is np.lib.npyio.NpzFile:
        keys = list(image.keys())
        assert len(keys) == 1
        image = image[keys[0]]
    if image_type is not None:
        image = image.astype(dtype=image_type)
    return Image.fromarray(image)


def load_image_in_known_formats(file: Path) -> Image:
    """
    Loads an image from a file in the given path. At the moment, this supports png and numpy files

    :param file: The path of the file to load.
    :return: the image
    """
    if is_numpy_file_path(file):
        return load_numpy_image(file)
    elif is_png(file):
        return load_png(file)
    else:
        raise ValueError(f"Unsupported image file type for path {file}")


class Report(FPDF):
    def __init__(self, title: str = "Default Report", output_folder: str = "outputs"):
        super().__init__()
        report_folder = Path(output_folder) / title.lower().replace(" ", "_")
        report_folder.mkdir(exist_ok=True, parents=True)

        self.title = title
        self.report_folder = report_folder

    def add_break(self, num_lines: int = 1):
        # num lines: The number of blank lines to add
        for _ in range(num_lines):
            self.ln()

    def add_title_header(self, title_str: str):
        self.set_table_style(font_style="B", font_size=18)
        self.cell(txt=title_str, ln=1, align="C", center=True)
        self.set_table_style()

    def add_header(self, heading_str: str):
        self.set_table_style(font_style="B", font_size=16)
        self.cell(txt=heading_str, ln=1)
        self.set_table_style()

    def set_table_style(self,
                        fill_color: Tuple[int, int, int] = (255, 255, 255),
                        text_color: int = 0,
                        line_color: Tuple[int, int, int] = (0, 0, 0),
                        line_width: float = 0.3,
                        font_style: str = "",
                        font_family: str = "helvetica",
                        font_size: int = 0) -> None:
        # set the table style, including background, font and line colours, line width and font style
        self.set_fill_color(*fill_color)
        self.set_text_color(text_color)
        self.set_draw_color(*line_color)
        self.set_line_width(line_width)  # type: ignore
        self.set_font(style=font_style, family=font_family, size=font_size)

    def _populate_table_rows(self, headers: List[str], data: List[List[str]], col_widths: List[float],
                             table_line_height: int, alternate_fill=True, border="LR"):
        # Data
        fill = False
        for row in data:
            for i in range(len(headers)):
                row_data = row[i]
                row_text = str(row_data) if not isinstance(row_data, str) else row_data
                self.cell(col_widths[i], table_line_height, txt=row_text, border=border, ln=0, align="L", fill=fill)
            self.ln()
            # change fill for next line
            fill = not fill if alternate_fill else False

        # self.cell(sum(w), 0, "", "T", ln=1)
        self.add_break(num_lines=1)

    # @staticmethod
    # def _update_precision(data: List[List[str]], headers: List[str], precision: Dict[str, int]):
    #     df = pd.DataFrame(data)
    #     df.columns = headers
    #     for col, desired_precision in precision.items():
    #         df[col] = df[col].round(desired_precision)
    #     return df.values.tolist()

    def add_table(self, headers: List[str] = None, data: Union[pd.DataFrame, List[List[str]]] = None,
                  data_path: Path = None, header_color=(153, 204, 204), header_font_color=0,
                  header_line_color=(0, 0, 0), header_line_width=0.3, body_line_color=(0, 0, 0), body_line_width=0.3,
                  alternate_fill=True, alternate_fill_color=(229, 229, 229), cell_precision: Dict[str, int] = None,
                  col_distribution="fitted", font_size=0, table_width=None):

        self.set_table_style(fill_color=header_color, text_color=header_font_color, font_style="B",
                             line_color=header_line_color, line_width=header_line_width,
                             font_size=font_size)

        table = Table(data=data, data_path=data_path, headers=headers, font_size_pt=self.font_size_pt,
                      table_width=table_width or self.epw)

        if cell_precision:
            table.update_precision(cell_precision)

        col_widths = table.get_column_widths(col_distribution=col_distribution)
        table_line_height = 7
        # padd all haders to same length as hack to make

        max_header_length = max([len(x) for x in headers])
        padded_headers = [h + ' ' * (max_header_length - len(h)) for h in headers]
        for i, (width, header_text) in enumerate(zip(col_widths, padded_headers)):
            self.cell(w=width, h=7, txt=header_text, align="L", fill=True)
            # ln = 1 if i == len(headers)-1 else 3
            # self.multi_cell(w=width, h=table_line_height, txt=header_text, fill=True, ln=ln)

        self.ln()
        self.set_table_style(fill_color=alternate_fill_color, line_color=body_line_color, line_width=body_line_width,
                             font_size=self.font_size_pt)
        data = table.to_list()
        self._populate_table_rows(headers, data, col_widths, table_line_height, alternate_fill=alternate_fill,
                                  border=None)

    def add_table_from_dataframe(self, df: pd.DataFrame, cols_to_print: Optional[List[str]] = None,
                                 print_index: bool = False, override_headers: List[str] = None):

        if cols_to_print:
            df = df[cols_to_print]
            headers = cols_to_print
        else:
            headers = df.columns.tolist()

        if print_index:
            # df = df.reset_index()
            index_col_name = "index"
            df.insert(0, index_col_name, np.arange(1, len(df) + 1))
            # Add the new index column to the header list
            headers = [index_col_name] + headers

        if override_headers:
            headers = override_headers

        # read headers from table
        data = df.to_numpy().tolist()
        assert len(headers) == len(data[0])
        self.add_table(headers, data)

    # def read_table_from_file(self, data_path: Path) -> pd.DataFrame:
    #     suffix = data_path.suffix
    #     if suffix == ".csv":
    #         df = pd.read_csv(data_path)
    #     elif suffix in [".xls", ".xlsx"]:
    #         df = pd.read_excel(data_path)
    #     else:
    #         # TODO: load more file types
    #         # with open(data_path, "r") as f_path:
    #         #     txt = f_path.read()
    #         #     lines = txt.split()
    #         raise ValueError(f"Can only read data from .csv, .xls or .xlsx files. Found {suffix}")
    #     return df

    def add_table_from_file(self, data_path: Path):
        df = self.read_table_from_file(data_path)
        self.add_table_from_dataframe(df)

    def save_report(self) -> Path:
        report_path = (self.report_folder / self.title.replace(" ", "_").lower()).with_suffix(".pdf")
        self.output(str(report_path))
        return report_path

    def add_line_chart_from_file(self, data_path: Path, x_col: str, y_col: str, chart_title: str = "",
                                 image_height: int = 0, image_width: int = 0, alt_text: str = "") -> None:
        df = self.read_table_from_file(data_path)
        x = df[x_col].values
        y = df[y_col].values
        self.add_line_chart(x, y, chart_title=chart_title, x_label=x_col, y_label=y_col, image_height=image_height,
                            image_width=image_width, alt_text=alt_text)

    def add_image(self, image_path):
        image: Image = load_image_in_known_formats(image_path)
        self.image(image)
        image.close()

    def add_image_gallery(self, image_paths: List[str], mode="RGB", border=0):
        images = [Image.open(img_path).convert(mode) for img_path in image_paths]
        widths, heights = [], []
        for img in images:
            widths.append(img.size[0])
            heights.append(img.size[1])

        max_height = max(heights)
        total_width = sum(widths)
        if total_width > self.epw:
            # TODO: split across rows
            pass

        combined_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            combined_image.paste(im, (x_offset, 0))
            x_offset += im.size[0] + border

        self.image(combined_image)
        [image.close() for image in images]


class Table:
    def __init__(self, headers: Optional[List[str]] = None, data: Union[pd.DataFrame, List[List[str]]] = None,
                 data_path=None, font_size_pt=None, table_width=None):
        self.headers = headers
        # Calling the below may update headers attribute
        self.df = self._get_data(data=data, data_path=data_path)
        self.font_size_pt = font_size_pt
        self.table_width = table_width

    def _get_data(self, data=None, data_path=None) -> pd.DataFrame:
        """
        If data is provided, will return that (in the form of a Pandas DataFrame if not already). Otherwise
        if a path to a data file is provided, load it as a Pandas DataFrame and return that

        :param data:
        :param data_path:
        :return:
        """
        if data is not None:
            if isinstance(data, pd.DataFrame):
                return data
            else:
                return pd.DataFrame(data, columns=self.headers)
        elif data_path is not None:
            return self._load_data_from_file(data_path)
        else:
            raise ValueError("One of data or data_path must be provided to create a data table")

    def _load_data_from_file(self, data_path: str) -> pd.DataFrame:
        if data_path.endswith(".txt"):
            raise NotImplementedError("Cant currently load from text file")
            # with open(data_path, 'r') as f_path:
            #      data = f_path.read()  --> convert
            # return pd.read_csv('file.txt', sep="\t")
        elif data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
            self.headers = df.columns
            return df
        elif data_path.endswith(".xls") or data_path.endswith(".xlsx"):
            df = pd.read_excel(data_path)
            self.headers = df.columns
            return df
        elif data_path.endswith(".json"):
            with open(data_path, "r") as f_path:
                # data = json.load(f_path)
                # self.headers = self.headers or data.keys()
                return pd.read_json(f_path)
        elif data_path.endswith(".jsonl"):
            with open(data_path, "r") as f_path:
                data = [json.loads(line) for line in f_path]
                # self.headers = self.headers or data[0].keys()
                # return data
            return pd.DataFrame.from_records(data)
        else:
            raise ValueError("Unexpected file format. Can't load data")

    def to_list(self) -> List[List[str]]:
        return self.df.values.tolist()

    # def to_dataframe(self, data: List[Any]):
    #     if isinstance(data, pd.DataFrame):
    #         logging.warning("Data is already a pandas DataFrame. Doing nothing")
    #         return data
    #     return pd.DataFrame(data, columns=self.headers)

    def update_precision(self, precision: Dict[str, int]):
        for col, desired_precision in precision.items():
            self.df[col] = self.df[col].round(desired_precision)

    def get_string_width_(self, x: str) -> int:
        if self.font_size_pt is None:
            raise ValueError("Cannot get string width if self.font_size_pt is None")
        x = str(x) if not isinstance(x, str) else x
        string_width = len(x) * self.font_size_pt
        # add additional leeway for special characters
        # r = re.compile(r'\[^a-zA-Z !@#$%&*_+-=|:";<>,./\(\)\[]\{\}']')
        # special_chars = r.findall(x)
        return np.floor(string_width)  # + 2 * len(special_chars))

    @staticmethod
    def _reduce_largest_col_with(header_widths: List[float]) -> List[float]:
        max_header_width = max(header_widths)
        largest_idx = header_widths.index(max_header_width)
        header_widths[largest_idx] = max_header_width * 0.8
        return header_widths

    def get_column_widths(self, col_distribution: str = "avg"):
        # TODO: col distr -> enum
        table_width = self.table_width
        if table_width is None:
            raise ValueError("Cannot determine optimal column widths if self.page_width is not known")
        avg_col_size = table_width / len(self.headers)
        if col_distribution == "avg":
            widths = [avg_col_size] * len(self.headers)
        elif col_distribution == "fitted":
            # TODO: figure out how to do multiline headers and then set col width equal to data width
            # num_chars = len(header_text)
            header_widths = [self.get_string_width_(header_text) for header_text in self.headers]
            data_widths = [self.get_string_width_(col_text) for col_text in self.df.iloc[0].tolist()]
            widths = data_widths  # [max(header_widths[i], data_widths[i]) for i in range(len(header_widths))]
            while sum(widths) > table_width:
                # print(widths)
                widths = self._reduce_largest_col_with(widths)
        else:
            raise ValueError("col_distribution must be one of avg or fitted")
        return widths


class Plot:
    def __init__(self, fig_folder, fig_title=None, n_rows=1, n_cols=1, figsize=(6.4, 4.8)):
        self.fig_folder = fig_folder
        self.fig_title = fig_title or "plot"
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        self.fig = fig
        # axes must be a list of lists of Axes objects
        if isinstance(axs, Axes):
            axes = [[axs]]
        elif isinstance(axs, np.ndarray):
            axes = axs.tolist()
            if isinstance(axes[0], Axes):
                axes = [axes]
        self.axs = axes
        assert isinstance(self.axs, List)
        assert isinstance(self.axs[0], List)
        assert isinstance(self.axs[0][0], Axes)
        self.current_axis = None

    @property
    def next_empty_axis(self):
        for ax_list in self.axs:
            for ax in ax_list:
                if not ax.has_data():
                    self.current_axis = ax
                    return ax
        raise ValueError("Trying to add more plots than originally specified")

    @staticmethod
    def label_plot(ax, chart_title: str = "", x_label: str = "", y_label: str = ""):
        if chart_title:
            ax.set_title(chart_title)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        return ax

    def save_chart(self) -> Path:
        plt.tight_layout()
        plot_path = (self.fig_folder / self.fig_title.lower().replace(" ", "_")).with_suffix(".png")
        self.fig.savefig(plot_path)
        return plot_path

    def add_line(self, x, y, chart_title, x_label, y_label, superimpose=False):
        if self.current_axis is None or superimpose is False:
            ax = self.next_empty_axis
        else:
            ax = self.current_axis

        ax.plot(x, y)
        self.label_plot(ax, chart_title=chart_title, x_label=x_label, y_label=y_label)

    def add_pr_curve(self, model_outputs, labels, plot_kwargs=None, superimpose=False):
        if self.current_axis is None or superimpose is False:
            ax = self.next_empty_axis
        else:
            ax = self.current_axis

        plot_kwargs = {} if plot_kwargs is None else {}

        fpr, tpr, thresholds = roc_curve(labels, model_outputs)
        ax.plot(fpr, tpr, **plot_kwargs)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR Curve")
        return ax

    def add_roc_curve(self, model_outputs, labels, plot_kwargs=None, superimpose=False):
        if self.current_axis is None or superimpose is False:
            ax = self.next_empty_axis
        else:
            ax = self.current_axis

        plot_kwargs = {} if plot_kwargs is None else {}

        precision, recall, thresholds = precision_recall_curve(labels, model_outputs)
        ax.plot(recall, precision, **plot_kwargs)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        return ax

    def plot_scores_and_summary(self, labels, model_outputs,
                                scoring_fn: Callable[[Tuple[List[Any], List[Any]]], Tuple[np.ndarray, np.ndarray]],
                                interval_width: float = .8,
                                ax: Optional[Axes] = None) -> Tuple[List, Any]:
        if ax is None:
            ax = plt.gca()
        x_grid = np.linspace(0, 1, 101)
        interp_ys = []
        line_handles = []
        for index, (lbls, mdl_out) in enumerate(zip(labels, model_outputs)):
            x_values, y_values = scoring_fn(lbls, mdl_out)
            interp_ys.append(np.interp(x_grid, x_values, y_values))
            handle, = ax.plot(x_values, y_values, lw=1)
            line_handles.append(handle)

        interval_quantiles = [.5 - interval_width / 2, .5, .5 + interval_width / 2]
        y_lo, y_mid, y_hi = np.quantile(interp_ys, interval_quantiles, axis=0)
        h1 = ax.fill_between(x_grid, y_lo, y_hi, color='k', alpha=.2, lw=0)
        h2, = ax.plot(x_grid, y_mid, 'k', lw=2)
        summary_handle = (h1, h2)
        return line_handles, summary_handle

    def add_pr_roc_curve_crossval(self, labels, model_outputs, superimpose=True) -> None:

        def get_roc_xy(labels, model_outputs) -> Tuple[np.ndarray, np.ndarray]:
            fpr, tpr, thresholds = roc_curve(labels, model_outputs)
            return fpr, tpr

        def get_pr_xy(labels, model_outputs) -> Tuple[np.ndarray, np.ndarray]:
            precision, recall, thresholds = precision_recall_curve(labels, model_outputs)
            return recall[::-1], precision[::-1]  # inverted to be in ascending order

        if self.current_axis is None or superimpose is False:
            ax = self.next_empty_axis
        else:
            ax = self.current_axis

        interval_width = .8
        line_handles, summary_handle = self.plot_scores_and_summary(labels, model_outputs,
                                                                    scoring_fn=get_roc_xy, ax=ax,
                                                                    interval_width=interval_width)
        line_labels = [f"Split {split_index}" for split_index in range(len(labels))]
        ax.legend(line_handles + [summary_handle],
                  line_labels + [f"Median \u00b1 {50 * interval_width:g}%"])

        # plot PR curve
        self.plot_scores_and_summary(labels, model_outputs, scoring_fn=get_pr_xy, ax=self.next_empty_axis,
                                     interval_width=interval_width)

    def add_box_plot(self, data, labels, title="", x_label="", y_label="", superimpose=False):
        if self.current_axis is None or superimpose is False:
            ax = self.next_empty_axis
        else:
            ax = self.current_axis

        ax.boxplot(data, labels, patch_artist=True, showmeans=True, meanline=True)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.set_xticks()
        ax.grid()

    def custom_plot(self, plot_name, plot_args=(), plot_kwargs={}, title="", x_label="", y_label="", superimpose=False):
        if self.current_axis is None or superimpose is False:
            ax = self.next_empty_axis
        else:
            ax = self.current_axis

        if not hasattr(ax, plot_name):
            raise ValueError(f"Unrecognised attribute {plot_name} for class {type(ax)}")

        callable = getattr(ax, plot_name)
        callable(*plot_args, **plot_kwargs)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    def plot_dataframe(self, df: pd.DataFrame):
        ax = self.next_empty_axis
        ax.set_axis_off()
        ax.set_frame_on(False)

        ax.table(cellText=df.values, colLabels=df.keys())  # , loc='center')


class HTMLReport:
    def __init__(self, title: str = "Report", output_folder: str = "outputs"):
        self.report_title = title
        self.output_folder = output_folder

        report_folder = Path(output_folder)  # / title.lower().replace(" ", "_")
        report_folder.mkdir(exist_ok=True, parents=True)

        self.report_folder = report_folder
        self.report_path_html = report_folder / (title.lower().replace(" ", "_") + '.html')
        self.report_path_pdf = report_folder / (title.lower().replace(" ", "_") + '.pdf')
        self.report_html = ""
        self.template = ""
        self.template_path = self._create_template()

        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('/')
        )
        self.render_kwargs: Dict[str, Any] = {"title": title}

    @staticmethod
    def _remove_html_end(report_stream):
        return report_stream.replace("</p>\n</div>\n</body>\n</html>", "")

    def _create_template(self):
        template_path = self.report_folder / "template.html"
        template_path.touch(exist_ok=True)

        self.template += """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
<title> {{title}} </title>
</head>
<body>
<div class="container-fluid justify-content-center align-items-center">
<div class="container">
<h1> {{title}} </h1>
</div>
<p>
</p>
</div>
</body>
</html>"""

        # with open(template_path, "w+") as f_path:
        #
        #     f_path.write(self.template)
        return template_path

    def add_table(self, df: pd.DataFrame):

        # First update the template to expect a table
        # with open(self.template_path, "r+") as f_path:
        #     template = f_path.read()
        self.template = self._remove_html_end(self.template)

        table_key = "tables"
        while f"% for table in {table_key} %" in self.template:
            table_key += f"_{str(random.randint(0, 10))}"

        self.template += """<div class="container">
{% for table in """ + table_key + """ %}
{{ table.to_html(classes=["table table-striped w-auto"]) | safe }}
{% endfor %}
</div>
</p>
</div>
</body>
</html>"""
        # f_path.seek(0)
        # f_path.write(template)

        self.render_kwargs.update({table_key: [df]})

    def add_image(self, image_path: str):
        if image_path.startswith(str(self.report_folder)):
            img_path_html = Path(image_path).relative_to(self.report_folder)
        else:
            img_path_html = Path(image_path)

        image_path_pdf = Path(image_path)

        # with open(self.template_path, "r+") as f_path:
        #     report = f_path.read()
        self.template = self._remove_html_end(self.template)

        image_key_html = IMAGE_KEY_HTML
        image_key_pdf = IMAGE_KEY_PDF
        while f"% for table in {image_key_html} %" in self.template:
            random_int = f"_{str(random.randint(0, 10))}"
            image_key_html += random_int
            image_key_pdf += random_int

        self.template += """<div class="container">
{% for image_path in """ + image_key_html + """ %}
<img src={{image_path}} alt={{image_path}}>
{% endfor %}
</div>
</p>
</div>
</body>
</html>"""
        # f_path.seek(0)
        # f_path.write(report)

        self.render_kwargs.update({image_key_html: [img_path_html], image_key_pdf: [image_path_pdf]})

    def render(self, save_html=True) -> None:
        """
        Render the report
        """

        # Now render the report
        # subs = self.env.get_template(str(self.template_path)).render(
        #     **self.render_kwargs
        # )
        subs = self.env.from_string(self.template).render(**self.render_kwargs)
        self.report_html = subs
        # write the substitution to a file
        if save_html:
            with open(self.report_path_html, 'w') as f_path:
                f_path.write(subs)

    def to_pdf(self):
        # html = wsp.HTML(self.report_path)
        # html.write_pdf(str(self.report_path).replace(".html", ".pdf"))

        # To render images we need to replace the relative path for HTML rendering with the absolute path
        if IMAGE_KEY_PDF in self.render_kwargs:
            img_keys_pdf = [k for k in self.render_kwargs.keys() if k.startswith(IMAGE_KEY_PDF)]
            for img_key_pdf in img_keys_pdf:
                corresponding_key_html = img_key_pdf.replace("pdf", "html")
                assert corresponding_key_html in self.template
                self.template = self.template.replace(corresponding_key_html, img_key_pdf)

        # Now rerender the HTML (without overwriting the saved HTML file) and convert this updated HTML to PDF
        self.render(save_html=False)

        with open(self.report_path_pdf, "wb+") as f_path:
            pisa_status = pisa.CreatePDF(self.report_html, dest=f_path)

        # self.report_path_pdf.mkdir(exist_ok=True)
        # pypandoc.convert_file(str(self.report_path_html), 'pdf', outputfile=str(self.report_path_pdf), format='html',
        #                       extra_args=['--pdf-engine','xelatex'])


def get_data_from_tensorboard_log():
    # TODO
    pass


def initialize_report(report_title: str = "Default Report", output_folder: str = "reports") -> Report:
    report = Report(title=report_title, output_folder=output_folder)
    report.add_page()
    # Set the header style
    report.set_table_style(font_style="B", font_size=16)
    report.add_title_header(report_title)
    report.add_break(num_lines=1)
    # Reset to default page style
    report.set_table_style()
    report.set_title(report_title)
    return report


def parse_arguments(args: List[str]) -> Namespace:
    # Parse command line arguments for generating reports
    parser = ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="reports",
                        help="The folder in which to store generated reports")
    parser.add_argument("--report_title", type=str, default=None, help="The title of the report")
    args = parser.parse_args()
    return args
