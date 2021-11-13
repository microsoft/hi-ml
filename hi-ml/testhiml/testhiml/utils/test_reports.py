#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pickle
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import ruamel.yaml
from matplotlib.lines import Line2D
from ruamel.yaml.comments import CommentedMap as OrderedDict, CommentedSeq as OrderedList

from health_ml.utils.reports import HTMLReport, IMAGE_KEY_HTML, TABLE_KEY_HTML, REPORT_CONTENTS_KEY, ReportComponentKey


@pytest.fixture
def html_report(tmp_path: Path) -> HTMLReport:
    report_title = "Dummy HTML report"
    report_dir = tmp_path / report_title.replace(" ", "_")
    html_report = HTMLReport(title=report_title, output_folder=str(report_dir))
    return html_report


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    return pd.DataFrame({"A": list(range(20)), "B": [3.14159 * (r ** 2) for r in range(20)]})


def test_html_report_validate() -> None:
    html_report = HTMLReport()
    with pytest.raises(Exception) as e:
        html_report.validate()
        assert "report_html is missing the tag" in str(e.value)


def test_html_report_add_table(html_report: HTMLReport, dummy_df: pd.DataFrame, tmp_path: Path) -> None:
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    table_keys_before = [k for k in render_kwargs_before.keys() if TABLE_KEY_HTML in k]
    num_tables = len(table_keys_before)

    html_report.add_table(dummy_df)

    html_template_difference = html_report.template.replace(html_template_before, "")

    assert html_template_difference.count("table.to_html") == 1
    assert f"{TABLE_KEY_HTML}_{num_tables}" in html_report.render_kwargs
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()


def test_html_report_add_image(html_report: HTMLReport, dummy_df: pd.DataFrame) -> None:
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    image_paths_before = [k for k in render_kwargs_before.keys() if IMAGE_KEY_HTML in k]
    num_imgs_before = len(image_paths_before)

    dummy_df_cols = list(dummy_df.columns)
    dummy_df.plot(x=dummy_df_cols[0], y=dummy_df_cols[1], kind="scatter")
    fig_path = html_report.report_folder / "fig1.png"
    plt.savefig(fig_path)
    html_report.add_image(str(fig_path))

    html_template_difference = html_report.template.replace(html_template_before, "")

    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    # Expect another keyword argument for imagepaths with num_imgs_before in the key (because starts at 0)
    assert f"{IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()


def test_html_report_add_plot(html_report: HTMLReport, dummy_df: pd.DataFrame) -> None:
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    # calling add_plot creates an image file and henceforth treats the plot as an image
    image_paths_before = [k for k in render_kwargs_before.keys() if IMAGE_KEY_HTML in k]
    num_imgs_before = len(image_paths_before)
    dummy_df_cols = list(dummy_df.columns)

    fig, ax = plt.subplots(1, 1)
    ax.plot(dummy_df[[dummy_df_cols[0]]], dummy_df[[dummy_df_cols[1]]])
    ax.set_xlabel("Radius")
    ax.set_ylabel("Area")
    fig.suptitle("Area vs radius")

    # pass a matplotlib Figure object to add_plot and check the difference in the HTML report
    html_report.add_plot(fig=fig)
    html_template_difference = html_report.template.replace(html_template_before, "")
    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    # Expect another keyword argument for imagepaths with num_imgs_before in the key (because starts at 0)
    assert f"{IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()

    # pass a saved image path to add_plot and check the difference in the HTML report
    new_plot_path = html_report.report_folder / "new_plot.png"
    fig.savefig(new_plot_path)
    html_report.add_plot(plot_path=new_plot_path)
    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    # Expect another keyword argument for imagepaths with num_imgs_before in the key (because starts at 0)
    assert f"{IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs
    assert html_report.render_kwargs[f"{IMAGE_KEY_HTML}_{num_imgs_before}"][0] == Path("Area_vs_radius.png")
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()


def test_html_report_render(html_report: HTMLReport, dummy_df: pd.DataFrame) -> None:
    dummy_df.plot(x="A", y="B", kind="scatter")
    dummy_df.plot(x="A", y="B", kind="scatter")
    fig_path = html_report.report_folder / "fig1.png"
    plt.savefig(fig_path)
    html_report.add_image(str(fig_path))

    df2 = pd.DataFrame({"Shape": ["square", "circle", "triangle"], "colour": ["Red", "Blue", "Yellow"],
                        "A very very very very very very very very very very very very very very very long title": [
                            1, 2, 3]})
    html_report.add_table(df2)

    html_report.add_text("Area vs radius chart", tag_class="h3")

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
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()


def test_html_report_read_config(html_report: HTMLReport, dummy_df: pd.DataFrame, tmp_path: Path) -> None:
    html_template_before = html_report._remove_html_end(html_report.template)
    # write a report config
    table_path = tmp_path / "dummy_table.csv"
    dummy_df.to_csv(table_path)
    dummy_df_cols = list(dummy_df.columns)

    plt.plot(dummy_df[[dummy_df_cols[0]]], dummy_df[[dummy_df_cols[1]]])

    report_config = OrderedDict({
        REPORT_CONTENTS_KEY: OrderedList([
            (ReportComponentKey.TABLE.value, table_path)
        ])
    })
    report_config_path = tmp_path / "report_config.yml"
    with open(report_config_path, "w+") as f_path:
        ruamel.yaml.dump(report_config, f_path)

    report_config = html_report.read_config_yaml(report_config_path)
    assert list(report_config.keys()) == [REPORT_CONTENTS_KEY]
    assert len(report_config[REPORT_CONTENTS_KEY]) == 1
    assert report_config[REPORT_CONTENTS_KEY][0] == (ReportComponentKey.TABLE.value, table_path)

    html_report.add_yaml_contents_to_report(report_config)
    html_template_difference = html_report.template.replace(html_template_before, "")
    assert html_template_difference.count("table.to_html") == 1
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()


def test_load_plot_from_pickle(tmp_path: Path, dummy_df: pd.DataFrame) -> None:
    # first create a plot and pickle it
    dummy_df_cols = list(dummy_df.columns)
    fig, ax = plt.subplots()

    dummy_df.plot(x=dummy_df_cols[0], y=dummy_df_cols[1], kind="line", ax=ax)
    img_pickle_path = tmp_path / "plot.pkl"
    with open(img_pickle_path, "wb+") as f_path:
        pickle.dump(ax, f_path)

    # then load the plot and check it has the expected properties
    img = HTMLReport.load_plot_from_pickle(img_pickle_path)
    lines = list(img.get_lines())
    assert len(lines) == 1
    assert isinstance(lines[0], Line2D)
    img_x_min, img_y_min, img_x_max, img_y_max = img.dataLim.bounds
    x = dummy_df[[dummy_df_cols[0]]].values
    y = dummy_df[[dummy_df_cols[1]]].values
    min_x, max_x = min(x)[0], max(x)[0]
    min_y, max_y = min(y)[0], max(y)[0]
    assert img_x_min == min_x
    assert img_x_max == max_x
    assert img_y_min == min_y
    assert img_y_max == max_y
    plt.show()


def test_load_pickled_plots_onto_subplot(tmp_path: Path):
    plt_folder = tmp_path / "figures"
    plt_folder.mkdir(exist_ok=True)
    for i in range(5):
        fig, ax = plt.subplots()
        ax.plot(np.random.randint(0, 10, 10), np.random.randint(0, 10, 10))
        fig.savefig(str(plt_folder / f"fig_{i}.png"))

    fig, axs = HTMLReport.load_pickled_plots_onto_subplot(plt_folder)
    plt.show()


def test_load_imgs_onto_subplot(tmp_path: Path):
    plt_folder = tmp_path / "figures"
    plt_folder.mkdir(exist_ok=True)
    num_imgs = 5
    num_plot_cols = 2
    for i in range(num_imgs):
        fig, ax = plt.subplots()
        ax.plot(np.random.randint(0, 10, 10), np.random.randint(0, 10, 10))
        fig.savefig(str(plt_folder / f"fig_{i}.png"))

    fig = HTMLReport.load_imgs_onto_subplot(plt_folder, num_plot_columns=num_plot_cols)
    # the number of axes on the plot will be num_imgs if num imgs % 2 == 0, else num_imgs + 1
    # since we defined num_plot_cols as 2
    expected_num_axis = num_imgs if num_imgs % 2 == 0 else num_imgs + 1
    assert len(fig.axs) == expected_num_axis

    plt.show()
