#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap as OrderedDict, CommentedSeq as OrderedList

from health_ml.utils.reports import (HTMLReport, IMAGE_KEY_HTML, TABLE_KEY_HTML, REPORT_CONTENTS_KEY,
                                     ReportComponentKey)


@pytest.fixture
def html_report(tmp_path: Path) -> HTMLReport:
    report_title = "Dummy HTML report"
    report_dir = tmp_path / report_title.replace(" ", "_")
    html_report = HTMLReport(title=report_title, output_folder=str(report_dir))
    return html_report


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    return pd.DataFrame({"A": list(range(20)), "B": [3.14159 * (r ** 2) for r in range(20)]})


@pytest.fixture
def dummy_fig_folder(tmp_path: Path) -> Path:
    plt_folder = tmp_path / "figures"
    plt_folder.mkdir(exist_ok=True)
    num_imgs = 5
    for i in range(num_imgs):
        fig, ax = plt.subplots()
        ax.plot(np.random.randint(0, 10, 10), np.random.randint(0, 10, 10))
        fig.savefig(str(plt_folder / f"fig_{i}.png"))
    return plt_folder


def test_html_report_validate() -> None:
    html_report = HTMLReport()
    # HTML report is basic template at this point so validate should pass
    html_report.validate()

    # now remove the closing tags, and validate should fail
    html_report.template = html_report._remove_html_end(html_report.template)
    with pytest.raises(Exception) as e:
        html_report.validate()
        assert "report_html is missing the tag" in str(e.value)


def test_html_report_add_heading(html_report: HTMLReport) -> None:
    dummy_heading = "Heading"

    for invalid_level in [0, 6]:
        with pytest.raises(ValueError) as e:
            html_report.add_heading(dummy_heading, level=invalid_level)
            assert "Level must be an integer between 1 and 5" in str(e)

    level = 2
    html_template_before = html_report._remove_html_end(html_report.template)
    html_report.add_heading(dummy_heading, level)
    html_template_difference = html_report.template.replace(html_template_before, "")
    assert html_template_difference.count(f"<h{level}") == 1


def test_html_report_add_tables(html_report: HTMLReport, dummy_df: pd.DataFrame, tmp_path: Path) -> None:
    # assert that ValueError is raised if neither table_path nor table is provided
    with pytest.raises(ValueError) as e:
        html_report.add_tables()
        assert "One of table or table path must be provided" in str(e)

    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    table_keys_before = [k for k in render_kwargs_before.keys() if TABLE_KEY_HTML in k]
    num_tables = len(table_keys_before)

    html_report.add_tables(tables=[dummy_df])

    html_template_difference = html_report.template.replace(html_template_before, "")

    assert html_template_difference.count("table.to_html") == 1
    assert f"{TABLE_KEY_HTML}_{num_tables}" in html_report.render_kwargs
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()


def test_html_report_add_images(html_report: HTMLReport, dummy_df: pd.DataFrame) -> None:
    html_template_before = html_report._remove_html_end(html_report.template)
    render_kwargs_before = html_report.render_kwargs
    image_paths_before = [k for k in render_kwargs_before.keys() if IMAGE_KEY_HTML in k]
    num_imgs_before = len(image_paths_before)

    dummy_df_cols = list(dummy_df.columns)
    dummy_df.plot(x=dummy_df_cols[0], y=dummy_df_cols[1], kind="scatter")
    fig_path = html_report.report_folder / "fig1.png"
    plt.savefig(fig_path)
    html_report.add_images([fig_path])

    html_template_difference = html_report.template.replace(html_template_before, "")

    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    # Expect another keyword argument for imagepaths with num_imgs_before in the key (because starts at 0)
    assert f"{IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()


def test_html_report_add_images_encoded(html_report: HTMLReport, dummy_df: pd.DataFrame) -> None:
    dummy_df_cols = list(dummy_df.columns)
    dummy_df.plot(x=dummy_df_cols[0], y=dummy_df_cols[1], kind="scatter")
    fig_path = html_report.report_folder / "fig1.png"
    plt.savefig(fig_path)
    html_report.add_images([fig_path], base64_encode=True)

    assert "png;base64" in html_report.render_kwargs["IMAGEPATHSHTML_0"][0]


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

    # check that if the plot has no title, it is saved with today's date in its name. Note that we can't
    # check for exact datetime since the seconds may not match between here and when the function is called
    # so instead we check for a file starting with today's date
    expected_figname_start_no_title = f"plot_{datetime.now().strftime('%Y%m%d%H')}"
    assert len(list(html_report.report_folder.glob(f"{expected_figname_start_no_title}*.png"))) == 0

    # pass a matplotlib Figure object to add_plot and check the difference in the HTML report
    html_report.add_plot(fig=fig)

    # check that the figre file got created
    assert len(list(html_report.report_folder.glob(f"{expected_figname_start_no_title}*.png"))) == 1

    # check that the template got updated correctly
    html_template_difference = html_report.template.replace(html_template_before, "")
    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    # Expect another keyword argument for imagepaths with num_imgs_before in the key (because starts at 0)
    assert f"{IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()

    # Pass a plot with a title and ensure the filepath includes the title
    fig.suptitle("Area vs radius")
    expected_plot_path = html_report.report_folder / "Area_vs_radius.png"
    assert not expected_plot_path.exists()
    html_report.add_plot(fig=fig)
    assert expected_plot_path.exists()
    assert html_report.render_kwargs[f"{IMAGE_KEY_HTML}_{num_imgs_before+1}"][0] == "Area_vs_radius.png"

    # pass a saved image path to add_plot and check the difference in the HTML report
    plot_title = "new_plot.png"
    new_plot_path = html_report.report_folder / plot_title
    fig.savefig(str(new_plot_path))
    html_report.add_plot(plot_path=new_plot_path)
    # the difference between the templates after calling add_image should be a single HTML <img> tag
    assert html_template_difference.count("<img src=") == 1
    assert f"{IMAGE_KEY_HTML}_{num_imgs_before}" in html_report.render_kwargs
    assert html_report.render_kwargs[f"{IMAGE_KEY_HTML}_{num_imgs_before+2}"][0] == plot_title
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()


def test_html_report_render(html_report: HTMLReport, dummy_df: pd.DataFrame) -> None:
    dummy_df.plot(x="A", y="B", kind="scatter")
    dummy_df.plot(x="A", y="B", kind="scatter")
    fig_path = html_report.report_folder / "fig1.png"
    plt.savefig(fig_path)
    html_report.add_images([fig_path])

    df2 = pd.DataFrame({"Shape": ["square", "circle", "triangle"], "colour": ["Red", "Blue", "Yellow"],
                        "A very very very very very very very very very very very very very very very long title": [
                            1, 2, 3]})
    html_report.add_tables(tables=[df2])

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

    report_config_contents = OrderedDict({
        REPORT_CONTENTS_KEY: OrderedList([
            {ReportComponentKey.TYPE.value: ReportComponentKey.TABLE.value,
             ReportComponentKey.VALUE.value: table_path}
        ])
    })
    report_config_path = tmp_path / "report_config.yml"
    with open(report_config_path, "w+") as f_path:
        ruamel.yaml.dump(report_config_contents, f_path)

    report_config = html_report.read_config_yaml(report_config_path)
    assert list(report_config.keys()) == [REPORT_CONTENTS_KEY]
    assert len(report_config[REPORT_CONTENTS_KEY]) == 1
    report_contents_first_entry = report_config[REPORT_CONTENTS_KEY][0]
    assert report_contents_first_entry[ReportComponentKey.TYPE.value] == ReportComponentKey.TABLE.value
    assert report_contents_first_entry[ReportComponentKey.VALUE.value] == table_path

    html_report.add_yaml_contents_to_report(report_config)
    html_template_difference = html_report.template.replace(html_template_before, "")
    assert html_template_difference.count("table.to_html") == 1
    # validate the report to ensure it includes the minimum necessary tags
    html_report.validate()


class MockPath:
    def __init__(self) -> None:
        self.path = Path("gallery_image_0.png")

    def is_dir(self) -> bool:
        return False

    def relative_to(self, report_folder: Path) -> Path:
        return self.path


@pytest.fixture()
def mock_table_dir(tmp_path: Path) -> Path:
    mock_dir = tmp_path / "tables"
    mock_dir.mkdir()
    for i in range(3):
        mock_path = mock_dir / f"table{i}.csv"
        mock_path.touch()
    return mock_dir


@patch("pandas.read_csv")
def test_add_yaml_contents_to_report_tables(mock_read_csv: MagicMock, mock_table_dir: Path, html_report: HTMLReport
                                            ) -> None:
    mock_read_csv.return_value = "dummy_df"
    html_template_before = html_report._remove_html_end(html_report.template)

    # pass in yaml contents with a mock folder path containing 3 csv files and check that 3 table tags
    # get added to the report
    yaml_contents_with_table_dir = OrderedDict({
        REPORT_CONTENTS_KEY: OrderedList([
            {ReportComponentKey.TYPE.value: ReportComponentKey.TABLE.value,
             ReportComponentKey.VALUE.value: mock_table_dir}
        ])})

    html_report.add_yaml_contents_to_report(yaml_contents_with_table_dir)
    html_template_difference = html_report.template.replace(html_template_before, "")
    num_new_tables = len(list(mock_table_dir.iterdir()))
    assert html_template_difference.count(r"{% for table in") == num_new_tables
    html_template_before = html_report._remove_html_end(html_report.template)

    # Now add single path
    yaml_contents_with_table_path = OrderedDict({
        REPORT_CONTENTS_KEY: OrderedList([
            {ReportComponentKey.TYPE.value: ReportComponentKey.TABLE.value,
             ReportComponentKey.VALUE.value: next(mock_table_dir.iterdir())}
        ])})

    html_report.add_yaml_contents_to_report(yaml_contents_with_table_path)
    html_template_difference = html_report.template.replace(html_template_before, "")
    num_new_tables = 1
    assert html_template_difference.count(r"{% for table in") == num_new_tables


def test_add_yaml_contents_to_report_images(html_report: HTMLReport, dummy_fig_folder: Path) -> None:
    html_template_before = html_report._remove_html_end(html_report.template)

    # Now add image folder - first as a gallery
    yaml_contents_with_img_dir_gallery = OrderedDict({
        REPORT_CONTENTS_KEY: OrderedList([
            {ReportComponentKey.TYPE.value: ReportComponentKey.IMAGE_GALLERY.value,
             ReportComponentKey.VALUE.value: str(dummy_fig_folder)}
        ])})

    with patch.object(HTMLReport, "load_imgs_onto_subplot", return_value=plt.figure()):
        html_report.add_yaml_contents_to_report(yaml_contents_with_img_dir_gallery)

    html_template_difference = html_report.template.replace(html_template_before, "")
    num_new_imgs = 1
    assert html_template_difference.count(r"<img src") == num_new_imgs
    html_template_before = html_report._remove_html_end(html_report.template)

    # add image folder as separate images
    yaml_contents_with_img_dir = OrderedDict({
        REPORT_CONTENTS_KEY: OrderedList([
            {ReportComponentKey.TYPE.value: ReportComponentKey.IMAGE.value,
             ReportComponentKey.VALUE.value: str(dummy_fig_folder)}
        ])})

    with patch.object(HTMLReport, "load_imgs_onto_subplot", return_value=plt.figure()):
        html_report.add_yaml_contents_to_report(yaml_contents_with_img_dir)

    html_template_difference = html_report.template.replace(html_template_before, "")
    num_new_imgs = len(list(dummy_fig_folder.iterdir()))
    assert html_template_difference.count(r"<img src") == num_new_imgs
    html_template_before = html_report._remove_html_end(html_report.template)

    # Now add single image path
    yaml_contents_with_img_path = OrderedDict({
        REPORT_CONTENTS_KEY: OrderedList([
            {ReportComponentKey.TYPE.value: ReportComponentKey.IMAGE.value,
             ReportComponentKey.VALUE.value: str(next(dummy_fig_folder.iterdir()))}
        ])})

    html_report.add_yaml_contents_to_report(yaml_contents_with_img_path)
    html_template_difference = html_report.template.replace(html_template_before, "")
    num_new_imgs = 1
    assert html_template_difference.count(r"<img src") == num_new_imgs


def test_add_yaml_contents_to_report_text(html_report: HTMLReport) -> None:
    html_template_before = html_report._remove_html_end(html_report.template)
    num_existing_paragraphs = 0

    yaml_contents_with_text = OrderedDict({
        REPORT_CONTENTS_KEY: OrderedList([
            {ReportComponentKey.TYPE.value: ReportComponentKey.TEXT.value,
             ReportComponentKey.VALUE.value: "dummy_text"}
        ])})

    html_report.add_yaml_contents_to_report(yaml_contents_with_text)
    html_template_difference = html_report.template.replace(html_template_before, "")
    num_existing_paragraphs += 1
    assert html_template_difference.count(r"<p") == num_existing_paragraphs


def test_load_imgs_onto_subplot(dummy_fig_folder: Path) -> None:
    num_imgs = len(list(dummy_fig_folder.glob("*.png")))
    num_plot_cols = 2
    fig = HTMLReport.load_imgs_onto_subplot([dummy_fig_folder], num_plot_columns=num_plot_cols)
    # the number of axes on the plot should be the same as num_imgs
    assert len(fig.axes) == num_imgs

    # try with a number of columns greater than the number of images
    fig = HTMLReport.load_imgs_onto_subplot([dummy_fig_folder], num_plot_columns=2 * num_imgs)
    assert len(fig.axes) == num_imgs

    # expect error if zero cols requested
    with pytest.raises(ValueError) as e:
        fig = HTMLReport.load_imgs_onto_subplot([dummy_fig_folder], num_plot_columns=0)
        assert "Can't have less than one column in your plot" in str(e)


def test_add_image_gallery(html_report: HTMLReport, dummy_fig_folder: Path) -> None:
    expected_fig_path = html_report.report_folder / "gallery_image_0.png"
    assert not expected_fig_path.exists()
    html_template_before = html_report._remove_html_end(html_report.template)

    html_report.add_image_gallery([dummy_fig_folder])

    html_difference = html_report.template.replace(html_template_before, "")
    assert html_difference.count("<img src") == 1
    assert expected_fig_path.exists()
    html_report.validate()


@patch("azureml.core.Run")
def test_download_report_contents_from_aml(mock_run: MagicMock, html_report: HTMLReport, dummy_df: pd.DataFrame,
                                           dummy_fig_folder: Path, tmp_path: Path) -> None:
    num_children = 3
    table_path = tmp_path / "dummy_table.csv"
    dummy_df.to_csv(table_path)

    run_id = "run_id_123"
    report_contents = OrderedList([
        {ReportComponentKey.TYPE.value: ReportComponentKey.IMAGE.value,
         ReportComponentKey.VALUE.value: str(next(dummy_fig_folder.iterdir()))},
        {ReportComponentKey.TYPE.value: ReportComponentKey.IMAGE_GALLERY.value,
         ReportComponentKey.VALUE.value: str(dummy_fig_folder)},
        {ReportComponentKey.TYPE.value: ReportComponentKey.TABLE.value,
         ReportComponentKey.VALUE.value: str(table_path)}
    ])
    hyperdrive_hyperparam_name = "learning_rate"
    with patch("health_ml.utils.reports.get_aml_run_from_run_id") as mock_get_run:
        mock_run = MagicMock()
        mock_run.type = "hyperdrive"
        mock_get_run.return_value = mock_run

        with patch("health_ml.utils.reports.download_files_from_hyperdrive_children") as mock_download:
            # mock the behaviour of downloading a file for each of the child runs
            mock_download.return_value = ["dummy_path_{i}" for i in range(num_children)]

            updated_contents = html_report.download_report_contents_from_aml(run_id, report_contents,
                                                                             hyperdrive_hyperparam_name)

            mock_get_run.assert_called_once()
            assert mock_download.call_count == len(report_contents)

            assert len(updated_contents) == len(report_contents)
            initial_contents_first_type = report_contents[0][ReportComponentKey.TYPE.value]
            updated_contents_first_type = updated_contents[0][ReportComponentKey.TYPE.value]
            assert initial_contents_first_type == updated_contents_first_type
            initial_contents_first_value = report_contents[0][ReportComponentKey.VALUE.value]
            updated_contents_first_value = updated_contents[0][ReportComponentKey.VALUE.value]
            assert initial_contents_first_value != updated_contents_first_value
            assert len(updated_contents_first_value.split(",")) == num_children
            assert updated_contents_first_value.split(",")[0] == str(mock_download.return_value[0])


def test_zip_folder(html_report: HTMLReport, dummy_df: pd.DataFrame) -> None:
    dummy_df.plot(x="A", y="B", kind="scatter")
    dummy_df.plot(x="A", y="B", kind="scatter")
    fig_path = html_report.report_folder / "fig1.png"
    plt.savefig(fig_path)
    html_report.add_images([fig_path])

    df2 = pd.DataFrame({"Shape": ["square", "circle", "triangle"],
                        "colour": ["Red", "Blue", "Yellow"],
                        "Number ": [1, 2, 3]
                        })
    html_report.add_tables(tables=[df2])

    html_report.add_text("Area vs radius chart", tag_class="h3")

    html_report.render()

    # call zip folder method and check it works
    expected_zip_folder_path = html_report.zip_report_folder()
    assert expected_zip_folder_path.exists()

    # unzip folder and check contents
    extract_path = Path(str(html_report.report_folder) + "_extracted")
    zipfile.ZipFile(expected_zip_folder_path).extractall(path=extract_path)
    assert extract_path.is_dir()
    assert len(list(extract_path.glob("*.csv"))) == 0
    assert len(list(extract_path.glob("*.png"))) == 1
    assert len(list(extract_path.glob("*.html"))) == 2


def test_zip_nested_folder(dummy_df: pd.DataFrame, html_report: HTMLReport) -> None:
    report_folder = html_report.report_folder

    dummy_df.plot(x="A", y="B", kind="scatter")
    dummy_df.plot(x="A", y="B", kind="scatter")

    folder_paths = []
    num_nested_folders = 5
    for i in range(num_nested_folders):
        nested_folder = report_folder / str(i)
        nested_folder.mkdir()
        nested_fig_path = nested_folder / "fig.png"
        plt.savefig(nested_fig_path)
        folder_paths.append(nested_folder)

    html_report.add_images(folder_paths)
    html_report.render()

    expected_zip_folder_path = html_report.zip_report_folder()
    assert expected_zip_folder_path.exists()

    # unzip folder and check contents
    extract_path = Path(str(html_report.report_folder) + "_extracted")
    zipfile.ZipFile(expected_zip_folder_path).extractall(path=extract_path)
    assert extract_path.is_dir()

    for i in range(num_nested_folders):
        folder_path = extract_path / str(i)
        assert folder_path.is_dir()
        assert (folder_path / "fig.png").exists()
