#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Transform the templates into valid Python test code.
"""

from pathlib import Path
from typing import Dict

from jinja2 import Template

here = Path(__file__).parent.resolve()


def render_environment_yaml(environment_yaml_path: Path, version: str) -> None:
    """
    Rewrite the environment.yml template with version into a file at environment_yaml_path.

    :param environment_yaml_path: Where to save environment.yml.
    :param version: hi-ml package version.
    :return: None
    """
    environment_yaml_template = (here / 'simple' / 'environment.yml.template').read_text()

    t = Template(environment_yaml_template)

    if version:
        pip = f"""
  - pip:
    - hi-ml{version}
"""
    else:
        pip = ""

    options = {'pip': pip}

    r = t.render(options)
    environment_yaml_path.write_text(r)


def render_test_script(entry_script_path: Path, extra_options: Dict[str, str],
                       compute_cluster_name: str, environment_yaml_path: Path) -> None:
    """
    Rewrite the template with standard options, and extra options into a file at entry_script_path.

    :param entry_script_path: Where to save script file.
    :param extra_options: Extra options for rendering.
    :param compute_cluster_name: Compute cluster name for testing.
    :param environment_yaml_path: Path to environment.yml.
    :return: None
    """
    hello_world_template = (here / 'simple' / 'hello_world_template.txt').read_text()

    t = Template(hello_world_template)

    default_options = {}
    default_options['entry_script'] = "Path(sys.argv[0])"
    default_options['workspace_config_path'] = "WORKSPACE_CONFIG_JSON"
    default_options['compute_cluster_name'] = f'"{compute_cluster_name}"'
    default_options['conda_environment_file'] = f'Path("{str(environment_yaml_path)}")'
    default_options['pip_extra_index_url'] = '"http://test_my_pip_extra_index_url"'
    default_options['private_pip_wheel_path'] = 'Path("test_my_private_pip_wheel_path")'
    default_options['wait_for_completion'] = 'True'
    default_options['wait_for_completion_show_output'] = 'True'

    all_options = dict(default_options, **extra_options)

    r = t.render(all_options)
    entry_script_path.write_text(r)
