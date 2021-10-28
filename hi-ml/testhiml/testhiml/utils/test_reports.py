import matplotlib.pyplot as plt
import sys
from argparse import Namespace

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from PIL import Image

from health_ml.utils.reports import reports as report_util


@pytest.mark.parametrize("args, attr_name, attr_val", [
    (["", "--output_folder", "report_folder"], "output_folder", "report_folder"),
    ([""], "output_folder", "reports"),
    # ([], "", "")
])
def test_create_args(args, attr_name, attr_val):
    with patch.object(sys, 'argv', args):
        args = report_util.parse_arguments(sys.argv[1:])
        assert hasattr(args, attr_name)
        assert getattr(args, attr_name) == attr_val


# def test_add_table_from_dataframe(tmp_path: Path):
#     df = pd.DataFrame({
#         "X": [1, 2, 3],
#         "Y": ["a", "b", "c"]
#     })
#     report_title = "My Report"
#     report_output_folder = tmp_path / "reports"
#     report = report_util.initialize_report(report_title=report_title, output_folder= report_output_folder)
#
#     report.add_table_from_dataframe(df)
#
#     report_path = report.save_report()
#     stripped_title = report_title.lower().replace(" ", "_")
#     expected_report_path = report_output_folder / stripped_title / (stripped_title + ".pdf")
#
#     assert report_path == expected_report_path
#     assert expected_report_path.exists()
#     assert expected_report_path.stat().st_size > 0


@pytest.mark.parametrize("string_to_test, font_size, expected_width", [
    # ("abc", 10, 30),
    ("16/02/2015", 5, 50)
])
def test_get_string_width_(string_to_test: str, font_size: int, expected_width: int):
    report = report_util.Report()
    report.set_font_size(font_size)
    assert report.get_string_width_(string_to_test) == expected_width


# -------------------
# GENERATE DUMMY REPORT
# -------------------

def dummy_table(report):
    headers = ["Experiment", "Epochs", "loss", "Avg ROC AUC"]
    data = [[1, 150, 5, 0.8], [2, 150, 4.9, 0.91]]
    df = pd.DataFrame(data=data, columns=headers)
    report.add_table_from_dataframe(df)


def dummy_performers_table(report):
    headers = ["Top 5 false positives"]
    data = [['Id 11 score 1.0'], ['Id 2 score 0.9'], ['Id 3 score 0.89'], ['Id 4 score 0.85']]
    report.add_table(headers, data, header_color=(255, 255, 255), alternate_fill=False, body_line_width=0.0)


def dummy_line_subplot(report):
    x = [1, 2, 3]
    y = [0.7, 0.77, 0.8]
    chart_title = "Line chart"
    x_label = "X"
    y_label = "Y"
    alt_text = ""
    image_width = 200
    image_height = 150

    plot = report_util.Plot(n_cols=2, n_rows=2, figsize=(12.8, 9.0), fig_folder=report.report_folder,
                            fig_title=chart_title)

    # hack since plots always seem stretched:
    # image_height = 0.8 * image_width if image_height == image_width else image_height

    plot.add_line(x, y, "plot 1", x_label, y_label)
    plot.add_line(x, y, "plot 2", x_label, y_label)
    plot.add_line(x, y, "plot 3", x_label, y_label)
    plot.add_line(x, y, "plot 4", x_label, y_label)

    plot_path = plot.save_chart()

    report.image(str(plot_path), x=None, y=None, h=image_height, w=image_width, alt_text=alt_text)


def dummy_pr_roc_curves(report):
    model_outputs = np.random.randint(0, 2, size=200)
    labels = np.random.randint(0, 2, size=200)
    chart_title = "PR ROC"
    image_width = 200
    image_height = 100
    alt_text = "ROC PR plots"

    plot = report_util.Plot(n_cols=2, figsize=(12.8, 6.0), fig_folder=report.report_folder, fig_title=chart_title)

    plot.add_roc_curve(model_outputs, labels)
    plot.add_pr_curve(model_outputs, labels)

    plot_path = plot.save_chart()

    report.image(plot_path, x=None, y=None, h=image_height, w=image_width, alt_text=alt_text)


def dummy_pr_roc_curves_crossval(report):
    model_outputs = np.random.randint(0, 2, size=[5, 200])
    labels = np.random.randint(0, 2, size=[5, 200])
    chart_title = "PR ROC Crossval"
    image_width = 200
    image_height = 100
    alt_text = "ROC PR plots Cross val"

    plot = report_util.Plot(n_cols=2, figsize=(12.8, 6.0), fig_folder=report.report_folder, fig_title=chart_title)

    plot.add_pr_roc_curve_crossval(model_outputs, labels)

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

    plot = report_util.Plot(figsize=(12.8, 6.0), fig_folder=report.report_folder, fig_title=chart_title)
    plot.add_box_plot(data, labels, title=chart_title, y_label="Dice")

    plot_path = plot.save_chart()
    report.image(plot_path, x=None, y=None, h=image_height, w=image_width, alt_text=alt_text)


def dummy_custom_plot(report):
    chart_title = "Scatter plot"
    image_width = 200
    image_height = 100
    alt_text = "scatter plot"
    plot = report_util.Plot(figsize=(12.8, 6.0), fig_folder=report.report_folder, fig_title=chart_title)

    for color in ['tab:blue', 'tab:orange', 'tab:green']:
        x = np.random.randint(0, 20, 20).tolist()
        y = np.random.randint(0, 20, 20).tolist()
        scale = 200.0 * np.random.rand(20)
        plot.custom_plot("scatter", (x, y), {'c': color, 's': scale, 'label': color},
                         x_label="X", y_label="Y", title="My chart", superimpose=True)

    plot_path = plot.save_chart()
    report.image(plot_path, x=None, y=None, h=image_height, w=image_width, alt_text=alt_text)


def dummy_image(output_folder, report):
    img_path = Path(output_folder) / f"img_00.png"
    img_array = np.random.randint(0, 255, [200, 200]).astype(np.uint8)
    np.save(str(str(img_path).split(".")[0]), img_array)
    im = Image.fromarray(img_array)
    im.save(img_path, "PNG")

    report.add_image(img_path)


def dummy_gallery(output_folder, report):
    img_paths = []
    for i in range(4):
        img_path = Path(output_folder) / f"img_{i}.png"
        img_array = np.random.randint(0, 255, [150, 150]).astype(np.uint8)
        np.save(str(str(img_path).split(".")[0]), img_array)
        im = Image.fromarray(img_array)
        im.save(img_path, "PNG")
        img_paths.append(str(img_path))

    report.add_image_gallery(img_paths, border=2)


def dummy_df_plot(output_folder, report):
    chart_title = "Dataframe"
    image_width = 200
    image_height = 100
    alt_text = "dataframe"
    plot = report_util.Plot(figsize=(12.8, 6.0), fig_folder=report.report_folder, fig_title=chart_title)

    headers = ["Experiment", "Epochs", "loss", "Average ROC AUC"]
    data = [[1, 150, 5, 0.8], [2, 150, 4.9, 0.91]]
    df = pd.DataFrame(data=data, columns=headers)

    plot.plot_dataframe(df)
    plot_path = plot.save_chart()
    report.image(plot_path, x=None, y=None, h=image_height, w=image_width, alt_text=alt_text)


def generate_dummy_report(args: Namespace) -> None:
    # Generate PDF report
    report = report_util.initialize_report(report_title=args.report_title, output_folder=args.output_folder)
    report.set_font_size(8)
    report.cell(txt="A short description of this report", center=True, align="C")
    report.add_break(2)

    report.add_header("Table with formatting")
    dummy_table(report)
    report.add_break()

    report.add_header("Table with plain formatting")
    dummy_performers_table(report)
    report.add_break()

    report.add_header("Multiple charts in a subplot")
    dummy_line_subplot(report)
    report.add_break()

    report.add_header("PR & ROC plots")
    dummy_pr_roc_curves(report)
    report.add_break()

    report.add_header(("Box plots"))
    dummy_boxplot(report)
    report.add_break()

    report.add_header("Custom plot")
    dummy_custom_plot(report)
    report.add_break()

    report.add_header("Single image")
    dummy_image(args.output_folder, report)
    report.add_break()

    report.add_header("Gallery of mutiple images")
    dummy_gallery(args.output_folder, report)
    report.add_break()

    report.add_header("Cross val PR ROC plots")
    dummy_pr_roc_curves_crossval(report)
    report.add_break()

    dummy_df_plot(args.output_folder, report)
    report.add_break()

    report.cell(txt="Some text at the end of the report", ln=1)
    report.save_report()


def test_generate_dummy_report(tmp_path: Path):
    report_title = "Dummy report"

    args = ["", "--output_folder", str(tmp_path), "--report_title", report_title]
    with patch.object(sys, 'argv', args):
        args = report_util.parse_arguments(sys.argv[1:])
    generate_dummy_report(args)

    report_title_formatted = report_title.lower().replace(" ", "_")

    expected_files = [report_title_formatted + '.pdf', 'dice.png', 'line_chart.png', 'pr_roc.png', 'scatter_plot.png']
    for expected_file in expected_files:
        expected_file_path = tmp_path / report_title_formatted / expected_file
        assert expected_file_path.exists()


def test_generate_dummy_html_report(tmp_path: Path):
    report_title = "Dummy HTML report"
    report_dir = tmp_path / report_title.replace(" ", "_")
    html_report = report_util.HTMLReport(title=report_title, output_folder=str(report_dir))

    df = pd.DataFrame({"A": [1.23345, 12.456345, 7.345345345, 7.45234, 6.345234], "B": [2, 5, 6, 7, 8]})
    html_report.add_table(df)

    df.plot(x="A", y="B", kind="scatter")
    fig_path = report_dir / "fig1.png"
    plt.savefig(fig_path)
    html_report.add_image(str(fig_path))

    df2 = pd.DataFrame({"Shape": ["square", "circle", "triangle"], "colour": ["Red", "Blue", "Yellow"]})
    html_report.add_table(df2)

    html_report.render()

    html_report.to_pdf()

    assert html_report.report_path_html.exists()
    assert html_report.report_path_pdf.exists()