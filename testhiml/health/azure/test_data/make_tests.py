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

    options = {'hi_ml_version': version}

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

    options = {}

    options['entry_script'] = extra_options.get('entry_script', "Path(sys.argv[0])")
    options['compute_cluster_name'] = extra_options.get('compute_cluster_name', f'"{compute_cluster_name}"')
    options['conda_environment_file'] = extra_options.get('conda_environment_file',
                                                          f'Path("{str(environment_yaml_path)}")')
    options['workspace_config_path'] = extra_options.get('workspace_config_path', 'None')
    options['environment_variables'] = extra_options.get('environment_variables', 'None')
    options['input_datasets'] = extra_options.get('input_datasets', 'None')
    options['output_datasets'] = extra_options.get('output_datasets', 'None')
    options['wait_for_completion'] = extra_options.get('wait_for_completion', 'True')
    options['wait_for_completion_show_output'] = extra_options.get('wait_for_completion_show_output', 'True')
    options['args'] = extra_options.get('args', '')
    options['body'] = extra_options.get('body', '')

    r = t.render(options)
    entry_script_path.write_text(r)
