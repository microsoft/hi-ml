#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Transform the templates into valid Python test code.
"""

from pathlib import Path
from typing import Dict, Optional

from jinja2 import Template

from testazure.utils_testazure import himl_azure_root, DEFAULT_IGNORE_FOLDERS


here = Path(__file__).parent.resolve()


def render_environment_yaml(environment_yaml_path: Path, version: str, run_requirements: bool,
                            extra_options: Optional[Dict[str, str]] = None) -> None:
    """
    Rewrite the environment.yml template with version into a file at environment_yaml_path.

    :param environment_yaml_path: Where to save environment.yml.
    :param version: hi-ml-azure package version.
    :param run_requirements: True to include run_requirements.txt.
    :param extra_options: Extra options for template rendering, e.g. additional Conda channels and dependencies.
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

    if extra_options:
        if 'pip' in extra_options:
            for pckg in extra_options['pip']:
                pip += f"    - {pckg}\n"

        channels = ""
        if 'conda_channels' in extra_options:
            for channel in extra_options['conda_channels']:
                channels += f"   - {channel}\n"

        conda_deps = ""
        if 'conda_dependencies' in extra_options:
            for dep in extra_options['conda_dependencies']:
                conda_deps += f"  - {dep}\n"

        options.update({'channels': channels, 'conda_dependencies': conda_deps, 'pip': pip})

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
    default_options['imports'] = 'import sys'
    default_options['prequel'] = ''
    default_options['compute_cluster_name'] = f'"{compute_cluster_name}"'
    default_options['entry_script'] = "Path(sys.argv[0])"
    default_options['aml_workspace'] = 'None'
    default_options['workspace_config_file'] = workspace_config_file_arg
    default_options['snapshot_root_directory'] = 'here'
    default_options['conda_environment_file'] = f'Path("{str(environment_yaml_path.as_posix())}")'
    default_options['environment_variables'] = 'None'
    default_options['pip_extra_index_url'] = '""'
    default_options['private_pip_wheel_path'] = 'None'
    default_options['ignored_folders'] = str(DEFAULT_IGNORE_FOLDERS)
    default_options['default_datastore'] = '""'
    default_options['input_datasets'] = 'None'
    default_options['output_datasets'] = 'None'
    default_options['wait_for_completion'] = 'True'
    default_options['identity_based_auth'] = 'False'
    default_options['wait_for_completion_show_output'] = 'True'
    default_options['args'] = ''
    default_options['body'] = ''
    default_options["tags"] = '{}'
    default_options["display_name"] = ''
    default_options["strictly_aml_v1"] = 'True'
    default_options["submit_to_azureml"] = 'False'

    all_options = dict(default_options, **extra_options)

    r = t.render(all_options)
    entry_script_path.write_text(r)


if __name__ == '__main__':
    test_folder = here / "test_make_tests"
    test_folder.mkdir(exist_ok=True)
    render_environment_yaml(test_folder / "environment1.yml", ">=3.14", False)
    render_environment_yaml(test_folder / "environment2.yml", "", True)
    render_environment_yaml(test_folder / "environment3.yml", "", False)

    render_test_script(test_folder / "test1.py", {}, "demo_cluster", test_folder / "environment1.yml")

    from uuid import uuid4
    message_guid = uuid4().hex

    extra_options: Dict[str, str] = {
        'prequel': """
    some_variable = "foo"
        """,
        'environment_variables': f"{{'message_guid': '{message_guid}'}}",
        'pip_extra_index_url': "'https://test.pypi.org/simple/'",
        'private_pip_wheel_path': "'demo_private_wheel.whl'",
        'ignored_folders': '[".config", ".mypy_cache", "hello_world_output"]',
        'default_datastore': "'DEMO_DEFAULT_DATASTORE'",
        'input_datasets': """[
            "input_blob1",
            DatasetConfig(name="input_blob2", datastore="datastore2"),
            DatasetConfig(name="input_blob3", datastore="datastore3",
                          target_folder="target_folder"),
            DatasetConfig(name="input_blob4", datastore="datastore4",
                          use_mounting=True),
        ]""",
        'output_datasets': """[
            "output_output1",
            DatasetConfig(name="output_blob2", datastore="datastore2"),
            DatasetConfig(name="output_blob3", datastore="datastore3",
                          use_mounting=False),
        ]""",
        'wait_for_completion': "False",
        'wait_for_completion_show_output': "False",
        'args': 'parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")',
        'body': 'print(f"The message was: {args.message}")',
        'imports': """
import json
import shutil"""
    }

    render_test_script(test_folder / "test2.py", extra_options, "demo_cluster", test_folder / "environment2.yml")
