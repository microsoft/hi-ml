#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Transform the templates into valid Python test code.
"""

import pathlib

from jinja2 import Template

here = pathlib.Path(__file__).parent.resolve()

hello_world_template = (here / 'simple' / 'hello_world_template.txt').read_text()

t = Template(hello_world_template)

configs = [
    ('hello_world.py', {'workspace_config': 'None', 'workspace_config_path': 'None'}),
    ('hello_world_config1.py', {'workspace_config': 'None', 'workspace_config_path': '"config.json"'})
]

for filename, config in configs:
    r = t.render(config)
    (here / 'simple' / filename).write_text(r)
