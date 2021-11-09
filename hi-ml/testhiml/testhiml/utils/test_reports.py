import matplotlib.pyplot as plt
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import nbformat as nbf
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from health_ml.utils.reports import reports as report_util
from health_ml.utils.reports.reports import HTMLReport


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


@pytest.fixture
def html_report(tmp_path: Path):
    report_title = "Dummy HTML report"
    report_dir = tmp_path / report_title.replace(" ", "_")
    html_report = report_util.HTMLReport(title=report_title, output_folder=str(report_dir))
    return html_report


def test_html_report_add_table(html_report: HTMLReport, tmp_path: Path):
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    table_keys_before = [k for k in render_kwargs_before.keys() if report_util.TABLE_KEY_HTML in k]
    num_tables = len(table_keys_before)

    df = pd.DataFrame({"A": [1.23345, 12.456345, 7.345345345, 7.45234, 6.345234], "B": [2, 5, 6, 7, 8]})
    html_report.add_table(df)

    html_template_difference = html_report.template.replace(html_template_before, "")

    assert html_template_difference.count("table.to_html") == 1
    assert f"{report_util.TABLE_KEY_HTML}_{num_tables}" in html_report.render_kwargs


def test_html_report_add_image(html_report: HTMLReport):
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    image_paths_before = [k for k in render_kwargs_before.keys() if report_util.IMAGE_KEY_HTML in k]
    num_imgs_before = len(image_paths_before)

    df = pd.DataFrame({"A": list(range(20)), "B": [3.14159 * (r ** 2) for r in range(20)]})
    df.plot(x="A", y="B", kind="scatter")
    fig_path = html_report.report_folder / "fig1.png"
    plt.savefig(fig_path)
    html_report.add_image(str(fig_path))

    html_template_difference = html_report.template.replace(html_template_before, "")

    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    # Expect another keyword argument for imagepaths with num_imgs_before in the key (because starts at 0)
    assert f"{report_util.IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs


def test_html_report_add_plot(html_report: HTMLReport):
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    # calling add_plot creates an image file and henceforth treats the plot as an image
    image_paths_before = [k for k in render_kwargs_before.keys() if report_util.IMAGE_KEY_HTML in k]
    num_imgs_before = len(image_paths_before)

    df = pd.DataFrame({"A": list(range(20)), "B": [3.14159 * (r ** 2) for r in range(20)]})
    fig, ax = plt.subplots(1, 1)
    ax.plot(df[["A"]], df[["B"]])
    ax.set_xlabel("Radius")
    ax.set_ylabel("Area")
    fig.suptitle("Area vs radius")

    # Pass a matplotlib Figure object to add_plot
    html_report.add_plot(fig=fig)

    html_template_difference = html_report.template.replace(html_template_before, "")

    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    # Expect another keyword argument for imagepaths with num_imgs_before in the key (because starts at 0)
    assert f"{report_util.IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs

    # pass a save image path to add_plot
    new_plot_path = html_report.report_folder / "new_plot.png"
    fig.savefig(new_plot_path)

    html_report.add_plot(plot_path=new_plot_path)

    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    # Expect another keyword argument for imagepaths with num_imgs_before in the key (because starts at 0)
    assert f"{report_util.IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs
    assert html_report.render_kwargs[f"{report_util.IMAGE_KEY_HTML}_{num_imgs_before}"][0] == Path("Area_vs_radius.png")


def test_html_report_render(html_report: HTMLReport):
    df = pd.DataFrame({"A": list(range(20)), "B": [3.14159 * (r ** 2) for r in range(20)]})
    df.plot(x="A", y="B", kind="scatter")
    df.plot(x="A", y="B", kind="scatter")
    fig_path = html_report.report_folder / "fig1.png"
    plt.savefig(fig_path)
    html_report.add_image(str(fig_path))

    df2 = pd.DataFrame({"Shape": ["square", "circle", "triangle"], "colour": ["Red", "Blue", "Yellow"],
                        "A very very very very very very very very very very very very very very very long title": [
                            1, 2, 3]})
    html_report.add_table(df2)

    df3 = pd.DataFrame({"A": list(range(20)), "B": [3.14159 * (r ** 2) for r in range(20)]})
    fig, ax = plt.subplots(1, 1)
    ax.plot(df3[["A"]], df3[["B"]])
    ax.set_xlabel("Radius")
    ax.set_ylabel("Area")
    fig.suptitle("Area vs radius")

    html_report.add_plot(fig=fig)

    html_report.render()

    # check that we have 2 image tags and 1 table tag in the rendered HTML
    rendered_report = html_report.report_html
    assert rendered_report.count("<img") == 2
    assert rendered_report.count("<table") == 1
    assert rendered_report.count("<body>") == rendered_report.count("</body>") == 1
    assert rendered_report.count("<html lang") == rendered_report.count("</html>") == 1
    assert html_report.report_path_html.exists()


@pytest.fixture
def nb_report(tmp_path: Path):
    report_title = "Dummy IPython report"
    report_dir = tmp_path / report_title.replace(" ", "_")
    nb_report = report_util.JupyterReport(title=report_title, output_folder=str(report_dir))

    nb_report.add_markdown("#Dummy report")
    nb_report.add_code_cell("""\
    %pylab inline
    hist(normal(size=2000), bins=50);""")

    return nb_report


@pytest.fixture
def dummy_img_filepath(tmp_path: Path):
    # Add an image
    a = np.random.randint(0, 255, (250, 250)).astype(np.uint8)
    img = Image.fromarray(a)
    # save this image outside of the report folder so we can check it gets correctly moved
    img_filepath = tmp_path / "dummy_image.png"
    img.save(img_filepath)
    return img_filepath

@pytest.fixture
def dummy_df():
    return pd.DataFrame({"A": list(range(20)), "B": [3.14159 * (r ** 2) for r in range(20)]})


def test_load_existing_notebook(nb_report: nbf.NotebookNode, tmp_path: Path):
    # First create the notebook path then attempt to read it back
    nb_report.render()

    read_notebook = report_util.JupyterReport(existing_notebook_path=nb_report.report_path)
    assert len(read_notebook.nb.cells) == 2
    assert read_notebook.nb.cells[0].get("cell_type") == "markdown"
    assert read_notebook.nb.cells[1].get("cell_type") == "code"


def test_add_table(nb_report: nbf.NotebookNode, dummy_df: pd.DataFrame):
    num_nb_cells_before = len(nb_report.nb_cells)

    nb_report.add_table(dummy_df)
    assert len(nb_report.nb_cells) == num_nb_cells_before + 1
    new_cell = nb_report.nb_cells[-1]
    assert new_cell.get("cell_type") == "code"
    assert "import pandas as pd" in new_cell.source
    assert "pd.DataFrame(" in new_cell.source


def test_add_image(nb_report: nbf.NotebookNode, dummy_img_filepath: Path):
    num_nb_cells_before = len(nb_report.nb_cells)

    # add the image to the report
    nb_report.add_image(str(dummy_img_filepath))
    assert len(nb_report.nb_cells) == num_nb_cells_before + 1
    new_cell = nb_report.nb_cells[-1]
    assert new_cell.get("cell_type") == "code"
    assert "from PIL import Image" in new_cell.source
    assert "Image.open(" in new_cell.source


def test_add_plot(nb_report: nbf.NotebookNode, dummy_df: pd.DataFrame):
    # expect filepath to start with today's date to be created when add_plot is called
    plot_startswith = str(nb_report.report_folder / f"plot_{datetime.now().strftime('%Y%m%d')}")
    plots_before = [str(p).startswith(plot_startswith) for p in nb_report.report_folder.iterdir()]
    num_nb_cells_before = len(nb_report.nb_cells)

    fig, ax = plt.subplots()
    ax.plot(dummy_df[["A"]], dummy_df[["B"]])
    ax.set_xlabel("radius")
    ax.set_ylabel('area')
    nb_report.add_plot(plt)

    plots = [str(p).startswith(plot_startswith) for p in nb_report.report_folder.iterdir()]
    assert len(plots) == len(plots_before) + 1
    assert len(nb_report.nb_cells) == num_nb_cells_before + 1

    # render the resport for visual inspection
    nb_report.render()
    new_cell = nb_report.nb.cells[-1]
    assert f"Image.open(\"{str(nb_report.report_folder)}" in new_cell.get("source")
    assert nb_report.report_path.exists()


def test_render(nb_report: nbf.NotebookNode, dummy_img_filepath: Path, dummy_df: pd.DataFrame):
    # Add a table
    assert not nb_report.report_path.exists()

    nb_report.add_table(dummy_df)

    nb_report.add_image(str(dummy_img_filepath))

    nb_report.render()
    assert nb_report.nb.cells == nb_report.nb_cells
    assert nb_report.report_path.exists()


def test_export_as_html(nb_report: nbf.NotebookNode, dummy_img_filepath: Path, dummy_df: pd.DataFrame):
    # Add a table
    assert not nb_report.report_path.exists()

    nb_report.add_table(dummy_df)

    nb_report.add_image(str(dummy_img_filepath))

    html = nb_report.export_html()
    assert nb_report.report_path_html.exists()
    assert html.count("<body>") == html.count("<body>") == 1


def test_export_and_remove_input_cells(nb_report: nbf.NotebookNode, dummy_img_filepath: Path):
    assert not nb_report.report_path.exists()
    df = pd.DataFrame({"A": list(range(20)), "B": [3.14159 * (r ** 2) for r in range(20)]})
    nb_report.add_table(df)

    nb_report.add_image(str(dummy_img_filepath))

    html = nb_report.export_and_remove_input_cells()
    assert nb_report.report_path_html.exists()
    assert html.count("<body>") == html.count("<body>") == 1


def test_add_himl_code_cell(nb_report: nbf.NotebookNode):

    himl_code = f"""\
from pathlib import Path
from health_azure import get_workspace
get_workspace(workspace_config_path=Path('config.json'))"""
    nb_report.add_code_cell(himl_code)
    html = nb_report.export_html()
    assert nb_report.report_path_html.exists()
