#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Transform the templates into valid Python test code.
"""

import pathlib

from jinja2 import Template

from health.azure.azure_util import RESOURCE_GROUP, SUBSCRIPTION_ID, WORKSPACE_NAME

here = pathlib.Path(__file__).parent.resolve()

hello_world_template = (here / 'simple' / 'hello_world_template.txt').read_text()

t = Template(hello_world_template)

configs = [
    ('hello_world.py', {
        'workspace_config': 'None',
        'workspace_config_path': 'None',
        'environment_variables': 'None'}),
    ('hello_world_config1.py', {
        'workspace_config': 'None',
        'workspace_config_path': '"config.json"',
        'environment_variables': 'None'}),
    ('hello_world_config2.py', {
        'workspace_config': f"""WorkspaceConfig(
        os.getenv("{WORKSPACE_NAME}", ""),
        os.getenv("{SUBSCRIPTION_ID}", ""),
        os.getenv("{RESOURCE_GROUP}", ""))""",
        'workspace_config_path': 'None',
        'environment_variables': 'None'})

]

for filename, config in configs:
    entry_script = here / 'simple' / filename
    config['entry_script'] = "os.sys.argv[0]"
    config['compute_cluster_name'] = 'os.getenv("COMPUTE_CLUSTER_NAME", "")'
    config['conda_environment_file'] = 'None'
    r = t.render(config)
    entry_script.write_text(r)
