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

from testazure.util import himl_azure_root

here = Path(__file__).parent.resolve()


def render_environment_yaml(environment_yaml_path: Path, version: str, run_requirements: bool) -> None:
    """
    Rewrite the environment.yml template with version into a file at environment_yaml_path.

    :param environment_yaml_path: Where to save environment.yml.
    :param version: hi-ml-azure package version.
    :param run_requirements: True to include run_requirements.txt.
    :return: None
    """
    environment_yaml_template = (here / 'simple' / 'environment.yml.template').read_text()

    t = Template(environment_yaml_template)

    if version:
        pip = f"""
  - pip:
    - hi-ml-azure{version}
"""
    elif run_requirements:
        pip = """
  - pip:
"""
        run_requirements_lines = (himl_azure_root() / 'run_requirements.txt').read_text().splitlines()
        for line in run_requirements_lines:
            pip += f"    - {line}\n"
    else:
        pip = ""

    options = {'pip': pip}

    r = t.render(options)
    environment_yaml_path.write_text(r)


def render_test_script(entry_script_path: Path, extra_options: Dict[str, str],
                       compute_cluster_name: str, environment_yaml_path: Path,
                       workspace_config_file_arg: str = "WORKSPACE_CONFIG_JSON") -> None:
    """
    Rewrite the template with standard options, and extra options into a file at entry_script_path.

    :param entry_script_path: Where to save script file.
    :param extra_options: Extra options for rendering.
    :param compute_cluster_name: Compute cluster name for testing.
    :param environment_yaml_path: Path to environment.yml.
    :param workspace_config_file_arg: The string that should be put into the script for the workspace_config_file
    argument.
    :return: None
    """
    hello_world_template = (here / 'simple' / 'hello_world_template.txt').read_text()

    t = Template(hello_world_template)

    default_options = {}
    default_options['prequel'] = ''
    default_options['compute_cluster_name'] = f'"{compute_cluster_name}"'
    default_options['entry_script'] = "Path(sys.argv[0])"
    default_options['aml_workspace'] = 'None'
    default_options['workspace_config_file'] = workspace_config_file_arg
    default_options['snapshot_root_directory'] = 'here'
    default_options['conda_environment_file'] = f'Path("{str(environment_yaml_path)}")'
    default_options['environment_variables'] = 'None'
    default_options['pip_extra_index_url'] = '""'
    default_options['private_pip_wheel_path'] = 'None'
    default_options['ignored_folders'] = '[".config", ".mypy_cache"]'
    default_options['default_datastore'] = '""'
    default_options['input_datasets'] = 'None'
    default_options['output_datasets'] = 'None'
    default_options['wait_for_completion'] = 'True'
    default_options['wait_for_completion_show_output'] = 'True'
    default_options['args'] = ''
    default_options['body'] = ''

    all_options = dict(default_options, **extra_options)

    r = t.render(all_options)
    entry_script_path.write_text(r)
