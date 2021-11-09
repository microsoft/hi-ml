#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pathlib import Path

from health_ml.utils.reports import HTMLReport, IMAGE_KEY_HTML, TABLE_KEY_HTML


@pytest.fixture
def html_report(tmp_path: Path):
    report_title = "Dummy HTML report"
    report_dir = tmp_path / report_title.replace(" ", "_")
    html_report = HTMLReport(title=report_title, output_folder=str(report_dir))
    return html_report


def test_html_report_add_table(html_report: HTMLReport, tmp_path: Path):
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    table_keys_before = [k for k in render_kwargs_before.keys() if TABLE_KEY_HTML in k]
    num_tables = len(table_keys_before)

    df = pd.DataFrame({"A": [1.23345, 12.456345, 7.345345345, 7.45234, 6.345234], "B": [2, 5, 6, 7, 8]})
    html_report.add_table(df)

    html_template_difference = html_report.template.replace(html_template_before, "")

    assert html_template_difference.count("table.to_html") == 1
    assert f"{TABLE_KEY_HTML}_{num_tables}" in html_report.render_kwargs


def test_html_report_add_image(html_report: HTMLReport):
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    image_paths_before = [k for k in render_kwargs_before.keys() if IMAGE_KEY_HTML in k]
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
    assert f"{IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs


def test_html_report_add_plot(html_report: HTMLReport):
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    # calling add_plot creates an image file and henceforth treats the plot as an image
    image_paths_before = [k for k in render_kwargs_before.keys() if IMAGE_KEY_HTML in k]
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
    assert f"{IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs

    # pass a save image path to add_plot
    new_plot_path = html_report.report_folder / "new_plot.png"
    fig.savefig(new_plot_path)

    html_report.add_plot(plot_path=new_plot_path)

    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    # Expect another keyword argument for imagepaths with num_imgs_before in the key (because starts at 0)
    assert f"{IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs
    assert html_report.render_kwargs[f"{IMAGE_KEY_HTML}_{num_imgs_before}"][0] == Path("Area_vs_radius.png")


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
