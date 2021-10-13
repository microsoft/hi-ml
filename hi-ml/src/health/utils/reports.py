import re
import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import auc, precision_recall_curve, recall_score, roc_auc_score, roc_curve

from fpdf import FPDF


def parse_arguments(args: List[str]) -> Namespace:
    # Parse command line arguments for generating reports
    parser = ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="reports",
                        help="The folder in which to store generated reports")
    args = parser.parse_args()
    return args


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
        self.set_table_style(font_style="B", font_size=16)
        self.cell(txt=title_str, ln=1, align="C", center=True)

    @staticmethod
    def _reduce_largest_col_with(header_widths: List[float]) -> List[float]:
        max_header_width = max(header_widths)
        largest_idx = header_widths.index(max_header_width)
        header_widths[largest_idx] = max_header_width * 0.8
        return header_widths

    def get_string_width_(self, x: str) -> int:
        x = str(x) if not isinstance(x, str) else x
        string_width = len(x) * self.font_size_pt
        # add additional leeway for special characters
        # r = re.compile(r'\[^a-zA-Z !@#$%&*_+-=|:";<>,./\(\)\[]\{\}']')
        # special_chars = r.findall(x)
        return np.ceil(string_width)  # + 2 * len(special_chars))

    def _get_column_widths(self, headers: List[str], data: List[List[str]], col_distribution: str = "avg"):
        # TODO: col distr -> enum
        page_width = self.epw
        avg_col_size = page_width / len(headers)
        if col_distribution == "avg":
            widths = [avg_col_size] * len(headers)
        elif col_distribution == "fitted":
            # num_chars = len(header_text)
            header_widths = [self.get_string_width_(header_text) for header_text in headers]
            data_widths = [self.get_string_width_(header_text) for header_text in data[0]]
            widths = [max(header_widths[i], data_widths[i]) for i in range(len(header_widths))]
            while sum(widths) > page_width:
                widths = self._reduce_largest_col_with(widths)
        else:
            raise ValueError("col_distribution must be one of avg or fitted")
        return widths

    def set_table_style(self,
                        fill_color: Tuple[int, int, int] = (255, 255, 255),
                        text_color: int = 0,
                        line_color: Tuple[int, int, int] = (0, 0, 0),
                        line_width: float = 0.3,
                        font_style: str = "",
                        font_family: str = "helvetica",
                        font_size: int = 10) -> None:
        # set the table style, including background, font and line colours, line width and font style
        self.set_fill_color(*fill_color)
        self.set_text_color(text_color)
        self.set_draw_color(*line_color)
        self.set_line_width(line_width)  # type: ignore
        self.set_font(style=font_style, family=font_family, size=font_size)

    def _populate_table_rows(self, headers: List[str], data: List[List[str]], col_widths: List[float],
                             table_line_height: int):
        # Data
        fill = False
        for row in data:
            for i in range(len(headers)):
                row_data = row[i]
                row_text = str(row_data) if not isinstance(row_data, str) else row_data
                self.cell(col_widths[i], table_line_height, txt=row_text, border="LR", ln=0, align="L", fill=fill)
            self.ln()
            # change fill for next line
            fill = not fill

        # self.cell(sum(w), 0, "", "T", ln=1)
        self.add_break(num_lines=1)

    def add_table(self, headers: List[str], data: List[List[str]]):

        self.set_table_style(fill_color=(0, 0, 255), font_style="B")

        # Colors, line width and bold font
        table_line_height = 7

        col_widths = self._get_column_widths(headers, data, col_distribution="fitted")

        for width, header_text in zip(col_widths, headers):
            # self.cell(w=width, h=7, txt=header_text, border=1, align="C", fill=True)
            self.cell(w=width, h=table_line_height, txt=header_text, fill=True)

        self.ln()
        self.set_table_style(fill_color=(224, 235, 255))

        self._populate_table_rows(headers, data, col_widths, table_line_height)

    def add_table_from_dataframe(self, df: pd.DataFrame):
        # read headers from table
        headers = list(df.columns)
        data = df.values.tolist()
        self.add_table(headers, data)

    def read_table_from_file(self, data_path: Path) -> pd.DataFrame:
        suffix = data_path.suffix
        if suffix == ".csv":
            df = pd.read_csv(data_path)
        elif suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(data_path)
        else:
            # TODO: load more file types
            # with open(data_path, "r") as f_path:
            #     txt = f_path.read()
            #     lines = txt.split()
            raise ValueError(f"Can only read data from .csv, .xls or .xlsx files. Found {suffix}")
        return df

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

    @property
    def current_axis(self):
        for ax_list in self.axs:
            for ax in ax_list:
                if not ax.has_data():
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

    def add_line(self, x, y, chart_title, x_label, y_label):
        ax = self.current_axis
        ax.plot(x, y)
        self.label_plot(ax, chart_title=chart_title, x_label=x_label, y_label=y_label)

    def add_pr_curve(self, model_outputs, labels, plot_kwargs=None):
        ax = self.current_axis
        plot_kwargs = {} if plot_kwargs is None else {}

        fpr, tpr, thresholds = roc_curve(labels, model_outputs)
        ax.plot(fpr, tpr, **plot_kwargs)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR Curve")
        return ax

    def add_roc_curve(self, model_outputs, labels, plot_kwargs=None):
        ax = self.current_axis
        plot_kwargs = {} if plot_kwargs is None else {}

        precision, recall, thresholds = precision_recall_curve(labels, model_outputs)
        ax.plot(recall, precision, **plot_kwargs)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        return ax

    def add_box_plot(self, data, labels, title="", x_label="", y_label=""):
        ax = self.current_axis

        ax.boxplot(data, labels, patch_artist=True, showmeans=True, meanline=True)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.set_xticks()
        ax.grid()


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


def dummy_table(report):
    headers = ["Experiment", "Epochs", "loss", "Average ROC AUC"]
    data = [[1, 150, 5, 0.8], [2, 150, 4.9, 0.91]]
    df = pd.DataFrame(data=data, columns=headers)
    report.add_table_from_dataframe(df)


def dummy_line_subplot(report):
    x = [1, 2, 3]
    y = [0.7, 0.77, 0.8]
    chart_title = "Line chart"
    x_label = "X"
    y_label = "Y"
    alt_text = ""
    image_width = 200
    image_height = 150

    plot = Plot(n_cols=2, n_rows=2, figsize=(12.8, 9.0), fig_folder=report.report_folder, fig_title=chart_title)

    # hack since plots always seem stretched:
    # image_height = 0.8 * image_width if image_height == image_width else image_height

    plot.add_line(x, y, "plot 1", x_label, y_label)
    plot.add_line(x, y, "plot 2", x_label, y_label)
    plot.add_line(x, y, "plot 3", x_label, y_label)
    plot.add_line(x, y, "plot 4", x_label, y_label)

    plot_path = plot.save_chart()

    report.image(plot_path, x=None, y=None, h=image_height, w=image_width, alt_text=alt_text)


def dummy_pr_auc_curves(report):
    model_outputs = np.random.randint(0, 2, size=200)
    labels = np.random.randint(0, 2, size=200)
    chart_title = "PR ROC"
    image_width = 200
    image_height = 100
    alt_text = "ROC PR plots"

    plot = Plot(n_cols=2, figsize=(12.8, 6.0), fig_folder=report.report_folder, fig_title=chart_title)

    plot.add_roc_curve(model_outputs, labels)
    plot.add_pr_curve(model_outputs, labels)

    plot_path = plot.save_chart()

    report.image(plot_path, x=None, y=None, h=image_height, w=image_width, alt_text=alt_text)


def dummy_boxplot(report):
    num_boxes = 5
    spreads = [np.random.rand(50) * 100 for _ in range(num_boxes)]
    centers = [np.ones(25) * 50 for _ in range(num_boxes)]
    flier_highs = [np.random.rand(10) * 100 + 100 for _ in range(num_boxes)]
    flier_lows = [np.random.rand(10) * -100 for _ in range(num_boxes)]
    data = [np.concatenate((spread, center, flier_high, flier_low)) for spread, center, flier_high, flier_low
            in zip(spreads, centers, flier_highs, flier_lows)]

    labels = np.arange(num_boxes).tolist()
    chart_title = "Dice"
    image_width = 200
    image_height = 100
    alt_text = "dice plot"

    plot = Plot(figsize=(12.8, 6.0), fig_folder=report.report_folder, fig_title=chart_title)
    plot.add_box_plot(data, labels, title=chart_title, y_label="Dice")

    plot_path = plot.save_chart()

    report.image(plot_path, x=None, y=None, h=image_height, w=image_width, alt_text=alt_text)


def generate_report(args: Namespace) -> None:
    # Generate PDF report
    report = initialize_report()

    report.cell(txt="A short description of this report", center=True, align="C")
    report.add_break(2)

    dummy_table(report)

    dummy_line_subplot(report)
    report.add_break()

    dummy_pr_auc_curves(report)
    report.add_break()

    dummy_boxplot(report)
    report.add_break()

    report.cell(txt="End of report", ln=1)
    report.save_report()


def main():
    args = parse_arguments(sys.argv[1:])
    generate_report(args)


if __name__ == "__main__":
    main()
