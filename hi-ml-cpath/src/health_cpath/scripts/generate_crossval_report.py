#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from matplotlib import pyplot as plt

from health_azure.utils import get_aml_run_from_run_id, get_workspace
from health_ml.utils.reports import HTMLReport
from health_cpath.utils.analysis_plot_utils import (add_training_curves_legend, plot_confusion_matrices,
                                                    plot_hyperdrive_roc_and_pr_curves,
                                                    plot_hyperdrive_training_curves)
from health_cpath.utils.output_utils import (AML_LEGACY_TEST_OUTPUTS_CSV, AML_TEST_OUTPUTS_CSV,
                                             AML_VAL_OUTPUTS_CSV)
from health_cpath.utils.report_utils import (collect_hyperdrive_metrics, collect_hyperdrive_outputs,
                                             child_runs_have_val_and_test_outputs, get_best_epoch_metrics,
                                             get_best_epochs, get_child_runs_hyperparams, get_hyperdrive_metrics_table,
                                             get_formatted_run_info, collect_class_info, get_max_epochs,
                                             download_hyperdrive_metrics_if_required)
from health_cpath.utils.naming import MetricsKey, ModelKey


def generate_html_report(parent_run_id: str, output_dir: Path,
                         workspace_config_path: Optional[Path] = None,
                         include_test: bool = False, overwrite: bool = False,
                         hyperdrive_arg_name: str = "crossval_index",
                         primary_metric: str = MetricsKey.AUROC) -> None:
    """
    Function to generate an HTML report of a Hyperdrive AML run (e.g., cross validation, different random seeds, ...).

    :param run_id: The parent Hyperdrive run ID.
    :param output_dir: Directory where to download Azure ML data and save the report.
    :param workspace_config_path: Path to Azure ML workspace config.json file.
        If omitted, will try to load default workspace.
    :param include_test: Include test results in the generated report.
    :param overwrite: Forces (re)download of metrics and output files, even if they already exist locally.
    :param hyperdrive_arg_name: Name of the Hyperdrive argument used for indexing the child runs.
        Default `crossval_index`.
    :param primary_metric: Name of the reference metric to optimise. Default `MetricsKey.AUROC`.
    """
    aml_workspace = get_workspace(workspace_config_path=workspace_config_path)
    parent_run = get_aml_run_from_run_id(parent_run_id, aml_workspace=aml_workspace)
    report_dir = output_dir / parent_run.display_name
    report_dir.mkdir(parents=True, exist_ok=True)

    report = HTMLReport(output_folder=report_dir)

    report.add_text(get_formatted_run_info(parent_run))

    report.add_heading("Azure ML metrics", level=2)

    # Download metrics from AML. Can take several seconds for each child run
    metrics_json = download_hyperdrive_metrics_if_required(parent_run_id, report_dir, aml_workspace,
                                                           overwrite=overwrite, hyperdrive_arg_name=hyperdrive_arg_name)

    # Get metrics dataframe from the downloaded json file
    metrics_df = collect_hyperdrive_metrics(metrics_json=metrics_json)

    hyperparameters_children = get_child_runs_hyperparams(metrics_df)
    max_epochs_dict = get_max_epochs(hyperparams_children=hyperparameters_children)
    best_epochs = get_best_epochs(metrics_df=metrics_df, primary_metric=f'{ModelKey.VAL}/{primary_metric}',
                                  max_epochs_dict=max_epochs_dict, maximise=True)

    # Add training curves for loss and AUROC (train and val.)
    render_training_curves(report, heading="Training curves", level=3,
                           metrics_df=metrics_df, best_epochs=best_epochs, report_dir=report_dir,
                           primary_metric=primary_metric)

    # Get metrics list with class names
    num_classes, class_names = collect_class_info(hyperparams_children=hyperparameters_children)

    base_metrics_list: List[str] = [MetricsKey.ACC, MetricsKey.AUROC, MetricsKey.AVERAGE_PRECISION,
                                    MetricsKey.COHENKAPPA]
    if num_classes > 1:
        base_metrics_list += [MetricsKey.ACC_MACRO, MetricsKey.ACC_WEIGHTED]
    else:
        base_metrics_list += [MetricsKey.F1, MetricsKey.PRECISION, MetricsKey.RECALL, MetricsKey.SPECIFICITY]

    base_metrics_list += class_names

    # Add tables with relevant metrics (val. and test)
    render_metrics_table(report,
                         heading=f"Validation metrics (best epoch based on maximum validation {primary_metric})",
                         level=3,
                         metrics_df=metrics_df, best_epochs=best_epochs,
                         base_metrics_list=base_metrics_list, metrics_prefix=f'{ModelKey.VAL}/')

    if include_test:
        render_metrics_table(report, heading="Test metrics", level=3,
                             metrics_df=metrics_df, best_epochs=None,
                             base_metrics_list=base_metrics_list, metrics_prefix=f'{ModelKey.TEST}/')

    # Get output data frames if available
    try:
        has_val_and_test_outputs = child_runs_have_val_and_test_outputs(parent_run)
        if has_val_and_test_outputs:
            output_filename_val = AML_VAL_OUTPUTS_CSV
            outputs_dfs_val = collect_hyperdrive_outputs(parent_run_id=parent_run_id, download_dir=report_dir,
                                                         aml_workspace=aml_workspace,
                                                         output_filename=output_filename_val, overwrite=overwrite,
                                                         hyperdrive_arg_name=hyperdrive_arg_name)
            if include_test:
                output_filename_test = AML_TEST_OUTPUTS_CSV if has_val_and_test_outputs else AML_LEGACY_TEST_OUTPUTS_CSV
                outputs_dfs_test = collect_hyperdrive_outputs(parent_run_id=parent_run_id, download_dir=report_dir,
                                                              aml_workspace=aml_workspace,
                                                              output_filename=output_filename_test, overwrite=overwrite,
                                                              hyperdrive_arg_name=hyperdrive_arg_name)

        if num_classes == 1:
            # Currently ROC and PR curves rendered only for binary case
            # TODO: Enable rendering of multi-class ROC and PR curves
            report.add_heading("ROC and PR curves", level=2)
            if has_val_and_test_outputs:
                # Add val. ROC and PR curves
                render_roc_and_pr_curves(report=report, heading="Validation ROC and PR curves", level=3,
                                         report_dir=report_dir,
                                         outputs_dfs=outputs_dfs_val,
                                         prefix=f'{ModelKey.VAL}_')
            if include_test:
                # Add test ROC and PR curves
                render_roc_and_pr_curves(report=report, heading="Test ROC and PR curves", level=3,
                                         report_dir=report_dir,
                                         outputs_dfs=outputs_dfs_test,
                                         prefix=f'{ModelKey.TEST}_')

        # Add confusion matrices for each fold
        report.add_heading("Confusion matrices", level=2)
        if has_val_and_test_outputs:
            # Add val. confusion matrices
            render_confusion_matrices(report=report, heading="Validation confusion matrices", level=3,
                                      class_names=class_names,
                                      report_dir=report_dir, outputs_dfs=outputs_dfs_val,
                                      prefix=f'{ModelKey.VAL}_')

        if include_test:
            # Add test confusion matrices
            render_confusion_matrices(report=report, heading="Test confusion matrices", level=3,
                                      class_names=class_names,
                                      report_dir=report_dir, outputs_dfs=outputs_dfs_test,
                                      prefix=f'{ModelKey.TEST}_')

    except ValueError as e:
        print(e)
        print("Since all expected output files were not found, skipping ROC-PR curves and confusion matrices.")

    # TODO: Add qualitative model outputs
    # report.add_heading("Qualitative model outputs", level=2)

    print(f"Rendering report to: {report.report_path_html.resolve()}")
    report.render()


def render_training_curves(report: HTMLReport, heading: str, level: int,
                           metrics_df: pd.DataFrame, best_epochs: Optional[Dict[int, int]],
                           report_dir: Path, primary_metric: str = MetricsKey.AUROC) -> None:
    """
    Function to render training curves for HTML reports.

    :param report: HTML report to perform the rendering.
    :param heading: Heading of the section.
    :param level: Level of HTML heading (e.g. sub-section, sub-sub-section) corresponding to HTML heading levels.
    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_hyperdrive_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param best_epochs: Dictionary mapping each hyperdrive child index to its best epoch.
    :param report_dir: Directory of the HTML report.
    :param primary_metric: Primary metric name. Default is AUROC.
    """
    report.add_heading(heading, level=level)
    metrics = {"loss_epoch", MetricsKey.AUROC.value, primary_metric}
    fig, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for i, metric in enumerate(metrics):
        plot_hyperdrive_training_curves(metrics_df, train_metric=f'{ModelKey.TRAIN}/{metric}',
                                        val_metric=f'{ModelKey.VAL}/{metric}',
                                        ylabel=metric, best_epochs=best_epochs, ax=axs[i])
    add_training_curves_legend(fig, include_best_epoch=True)
    training_curves_fig_path = report_dir / "training_curves.png"
    fig.savefig(training_curves_fig_path, bbox_inches='tight')
    report.add_images([training_curves_fig_path], base64_encode=True)


def render_metrics_table(report: HTMLReport, heading: str, level: int,
                         metrics_df: pd.DataFrame, best_epochs: Optional[Dict[int, int]],
                         base_metrics_list: List[str], metrics_prefix: str) -> None:
    """
    Function to render metrics table for HTML reports.

    :param report: HTML report to perform the rendering.
    :param heading: Heading of the section.
    :param level: Level of HTML heading (e.g. sub-section, sub-sub-section) corresponding to HTML heading levels.
    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_hyperdrive_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param base_metrics_list: List of metric names to include in the table.
    :param best_epochs: Dictionary mapping each hyperdrive child index to its best epoch.
    :param metrics_prefix: Prefix to add to the metrics names (e.g. `val`, `test`)
    """
    report.add_heading(heading, level=level)
    metrics_list = [metrics_prefix + metric for metric in base_metrics_list]
    if best_epochs:
        metrics_df = get_best_epoch_metrics(metrics_df, metrics_list, best_epochs)
    metrics_table = get_hyperdrive_metrics_table(metrics_df, metrics_list)
    report.add_tables([metrics_table])


def render_roc_and_pr_curves(report: HTMLReport, heading: str, level: int, report_dir: Path,
                             outputs_dfs: Dict[int, pd.DataFrame], prefix: str = '') -> None:
    """
    Function to render ROC and PR curves for HTML reports.

    :param report: HTML report to perform the rendering.
    :param heading: Heading of the section.
    :param level: Level of HTML heading (e.g. sub-section, sub-sub-section) corresponding to HTML heading levels.
    :param report_dir: Local directory where the report is stored.
    :param outputs_dfs: A dictionary of dataframes with the sorted hyperdrive child runs indices as keys.
    :param prefix: Prefix to add to the figures saved (e.g. `val`, `test`).
    """
    report.add_heading(heading, level=level)
    fig = plot_hyperdrive_roc_and_pr_curves(outputs_dfs, scores_column='prob_class1')
    roc_pr_curves_fig_path = report_dir / f"{prefix}roc_pr_curves.png"
    fig.savefig(roc_pr_curves_fig_path, bbox_inches='tight')
    report.add_images([roc_pr_curves_fig_path], base64_encode=True)


def render_confusion_matrices(report: HTMLReport, heading: str, level: int, class_names: List[str],
                              report_dir: Path, outputs_dfs: Dict[int, pd.DataFrame], prefix: str = '') -> None:
    """
    Function to render confusion matrices for HTML reports.

    :param report: HTML report to perform the rendering.
    :param heading: Heading of the section.
    :param level: Level of HTML heading (e.g. sub-section, sub-sub-section) corresponding to HTML heading levels.
    :param class_names: Names of classes.
    :param report_dir: Local directory where the report is stored.
    :param outputs_dfs: A dictionary of dataframes with the sorted hyperdrive child runs indices as keys.
    :param prefix: Prefix to add to the figures saved (e.g. `val`, `test`).
    """
    report.add_heading(heading, level=level)
    fig = plot_confusion_matrices(hyperdrive_dfs=outputs_dfs, class_names=class_names)
    confusion_matrices_fig_path = report_dir / f"{prefix}confusion_matrices.png"
    fig.savefig(confusion_matrices_fig_path, bbox_inches='tight')
    report.add_images([confusion_matrices_fig_path], base64_encode=True)


if __name__ == "__main__":
    """
    Usage example from CLI:
    python generate_hyperdrive_report.py \
    --run_id <insert AML run ID here> \
    --output_dir outputs \
    --include_test
    """

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--run_id', help="The parent Hyperdrive run ID.")
    parser.add_argument('--output_dir', help="Directory where to download Azure ML data and save the report.")
    parser.add_argument('--workspace_config', help="Path to Azure ML workspace config.json file. "
                                                   "If omitted, will try to load default workspace.")
    parser.add_argument('--include_test', action='store_true', help="Opt-in flag to include test results "
                                                                    "in the generated report.")
    parser.add_argument('--overwrite', action='store_true', help="Forces (re)download of metrics and output files, "
                                                                 "even if they already exist locally.")
    parser.add_argument("--hyper_arg_name", default="crossval_index",
                        help="Name of the Hyperdrive argument used for indexing the child runs.")
    parser.add_argument("--primary_metric", default=MetricsKey.AUROC, help="Name of the reference metric to optimise.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path.cwd() / "outputs"
    workspace_config = Path(args.workspace_config).resolve() if args.workspace_config else None

    print(f"Output dir: {Path(args.output_dir).resolve()}")
    if workspace_config is not None:
        if not workspace_config.is_file():
            raise ValueError(f"Specified workspace config file does not exist: {workspace_config}")
        print(f"Workspace config: {workspace_config}")

    generate_html_report(parent_run_id=args.run_id,
                         output_dir=Path(args.output_dir),
                         workspace_config_path=workspace_config,
                         include_test=args.include_test,
                         overwrite=args.overwrite,
                         hyperdrive_arg_name=args.hyper_arg_name,
                         primary_metric=args.primary_metric)
