
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import jinja2
import matplotlib.pyplot as plt
import pandas as pd

IMAGE_KEY_HTML = "IMAGEPATHSHTML"
TABLE_KEY_HTML = "TABLEKEYHTML"


class HTMLReport:
    """
    Create an HTML report which Azure ML is able to render. Note that the method convert_to_pdf will not
    apply bootstrap formatting. Very limited CSS properties are applied to the PDF
    (see [here](https://xhtml2pdf.readthedocs.io/en/latest/reference.html#supported-css-properties))
    """
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
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style type="text/css">
    @page body{
        size: A4 portrait;
        @frame header_frame {
            -pdf-frame-content: header_content;
            left: 50pt; width: 512pt; top: 50pt; height: 40pt;
        }
        @frame content_frame {
            left: 50pt; width: 512pt; top: 90pt; height: 632pt;
        }
        @frame footer_frame {
            -pdf-frame-content: footer_content;
            left: 50pt; width: 512pt; top: 772pt; height: 20pt;
        }
    }
    tbody td {
        text-align: center;
        vertical-align: middle}
    thead th {
        text-align: center;}
</style>
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

    def add_table(self, df: pd.DataFrame):

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

        self.render_kwargs.update({table_key: [df]})

    def add_image(self, image_path: str):
        if image_path.startswith(str(self.report_folder)):
            img_path_html = Path(image_path).relative_to(self.report_folder)
        else:
            img_path_html = Path(image_path)

        image_path_pdf = Path(image_path)

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
