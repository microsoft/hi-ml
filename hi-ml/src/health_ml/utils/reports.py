
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import jinja2
import ruamel.yaml
import matplotlib.pyplot as plt
import pandas as pd

IMAGE_KEY_HTML = "IMAGEPATHSHTML"
TABLE_KEY_HTML = "TABLEKEYHTML"
REPORT_CONTENTS_KEY = "report_contents"


class ReportComponentKey(Enum):
    IMAGE = "image"
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

    def validate(self):
        """
        For our definition the rendered HTML must contain exactly one open and closing tags doctype, head and body

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
    def _remove_html_end(report_stream):
        return report_stream.replace("</p>\n</div>\n</body>\n</html>", "")

    def _create_template(self):
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
</p>
</div>
</body>
</html>"""

        return template_path

    def add_text(self, text: str):
        self.template += f"<p>{text}</p>"

    def add_table(self, table: Optional[pd.DataFrame] = None, table_path: Optional[Path] = None):
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
<br>
</p>
</div>
</body>
</html>"""

        self.render_kwargs.update({table_key: [table]})

    def add_image(self, image_path: str):
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
<br>
</p>
</div>
</body>
</html>"""

        # Add these keys and paths to the keyword args for rendering later
        self.render_kwargs.update({image_key_html: [img_path_html]})

    def add_plot(self, plot_path: Optional[str] = None, fig: Optional[plt.Figure] = None):
        if fig is not None:
            # save the plot
            plot_title = fig._suptitle.get_text() or fig.texts[0].get_text()
            if len(plot_title) > 1:
                title = plot_title.replace(" ", "_")+ ".png"
            else:
                title = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            plot_path = self.report_folder / title
            fig.tight_layout()
            fig.savefig(plot_path, bbox_inches='tight', dpi=150)

        self.add_image(str(plot_path))

    def read_config_yaml(self, report_config_path: Path):
        assert report_config_path.suffix == ".yml", f"Expected a .yml file but found {report_config_path.suffix}"
        with open(report_config_path, "r") as f_path:
            yaml_contents = ruamel.yaml.load(f_path)

        report_contents = yaml_contents[REPORT_CONTENTS_KEY]
        for componenet_type, component_val in report_contents:
            if componenet_type == ReportComponentKey.TABLE.value:
                self.add_table(table_path=component_val)
            elif componenet_type == ReportComponentKey.IMAGE.value:
                self.add_image(component_val)
            elif componenet_type == ReportComponentKey.TEXT.value:
                self.add_text(component_val)
            else:
                raise ValueError("Key must either equal table or image")

        return yaml_contents

    def render(self, save_html=True) -> None:
        """
        Render the report
        """
        subs: str = self.env.from_string(self.template).render(**self.render_kwargs)
        self.report_html = subs
        # write the substitution to a file
        if save_html:
            with open(self.report_path_html, 'w') as f_path:
                f_path.write(subs)
