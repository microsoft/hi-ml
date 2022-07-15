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
                                                      plot_crossval_roc_and_pr_curves,
                                                      plot_crossval_training_curves)
from health_cpath.utils.output_utils import (AML_LEGACY_TEST_OUTPUTS_CSV, AML_TEST_OUTPUTS_CSV,
                                               AML_VAL_OUTPUTS_CSV)
from health_cpath.utils.report_utils import (collect_crossval_metrics, collect_crossval_outputs,
                                               crossval_runs_have_val_and_test_outputs, get_best_epoch_metrics,
                                               get_best_epochs, get_crossval_metrics_table, get_formatted_run_info,
                                               collect_class_info)
from health_cpath.utils.naming import MetricsKey, ModelKey


def generate_html_report(parent_run_id: str, output_dir: Path,
                         workspace_config_path: Optional[Path] = None,
                         include_test: bool = False, overwrite: bool = False) -> None:
    """
    Function to generate an HTML report of a cross validation AML run.

    :param run_id: The parent Hyperdrive run ID.
    :param output_dir: Directory where to download Azure ML data and save the report.
    :param workspace_config_path: Path to Azure ML workspace config.json file.
        If omitted, will try to load default workspace.
    :param include_test: Include test results in the generated report.
    :param overwrite: Forces (re)download of metrics and output files, even if they already exist locally.
    """
    aml_workspace = get_workspace(workspace_config_path=workspace_config_path)
    parent_run = get_aml_run_from_run_id(parent_run_id, aml_workspace=aml_workspace)
    report_dir = output_dir / parent_run.display_name
    report_dir.mkdir(parents=True, exist_ok=True)

    report = HTMLReport(output_folder=report_dir)

    report.add_text(get_formatted_run_info(parent_run))

    report.add_heading("Azure ML metrics", level=2)

    # Download metrics from AML. Can take several seconds for each child run
    metrics_df = collect_crossval_metrics(parent_run_id, report_dir, aml_workspace, overwrite=overwrite)
    best_epochs = get_best_epochs(metrics_df, f'{ModelKey.VAL}/{MetricsKey.AUROC}', maximise=True)

    # Add training curves for loss and AUROC (train and val.)
    render_training_curves(report, heading="Training curves", level=3,
                           metrics_df=metrics_df, best_epochs=best_epochs, report_dir=report_dir)

    # Get metrics list with class names
    num_classes, class_names = collect_class_info(metrics_df=metrics_df)

    base_metrics_list: List[str]
    if num_classes > 1:
        base_metrics_list = [MetricsKey.ACC, MetricsKey.AUROC]
    else:
        base_metrics_list = [MetricsKey.ACC, MetricsKey.AUROC, MetricsKey.PRECISION, MetricsKey.RECALL, MetricsKey.F1]

    base_metrics_list += class_names

    # Add tables with relevant metrics (val. and test)
    render_metrics_table(report, heading="Validation metrics (best epoch based on maximum validation AUROC)", level=3,
                         metrics_df=metrics_df, best_epochs=best_epochs,
                         base_metrics_list=base_metrics_list, metrics_prefix=f'{ModelKey.VAL}/')

    if include_test:
        render_metrics_table(report, heading="Test metrics", level=3,
                             metrics_df=metrics_df, best_epochs=None,
                             base_metrics_list=base_metrics_list, metrics_prefix=f'{ModelKey.TEST}/')

    has_val_and_test_outputs = crossval_runs_have_val_and_test_outputs(parent_run)

    # Get output data frames
    if has_val_and_test_outputs:
        output_filename_val = AML_VAL_OUTPUTS_CSV
        outputs_dfs_val = collect_crossval_outputs(parent_run_id=parent_run_id, download_dir=report_dir,
                                                   aml_workspace=aml_workspace,
                                                   output_filename=output_filename_val, overwrite=overwrite)
        if include_test:
            output_filename_test = AML_TEST_OUTPUTS_CSV if has_val_and_test_outputs else AML_LEGACY_TEST_OUTPUTS_CSV
            outputs_dfs_test = collect_crossval_outputs(parent_run_id=parent_run_id, download_dir=report_dir,
                                                        aml_workspace=aml_workspace,
                                                        output_filename=output_filename_test, overwrite=overwrite)

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

    # TODO: Add qualitative model outputs
    # report.add_heading("Qualitative model outputs", level=2)

    print(f"Rendering report to: {report.report_path_html.resolve()}")
    report.render()


def render_training_curves(report: HTMLReport, heading: str, level: int,
                           metrics_df: pd.DataFrame, best_epochs: Optional[Dict[int, int]],
                           report_dir: Path) -> None:
    """
    Function to render training curves for HTML reports.

    :param report: HTML report to perform the rendering.
    :param heading: Heading of the section.
    :param level: Level of HTML heading (e.g. sub-section, sub-sub-section) corresponding to HTML heading levels.
    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_crossval_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param best_epochs: Dictionary mapping each cross-validation index to its best epoch.
    :param report_dir: Directory of the HTML report.
    """
    report.add_heading(heading, level=level)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plot_crossval_training_curves(metrics_df, train_metric=f'{ModelKey.TRAIN}/loss_epoch',
                                  val_metric=f'{ModelKey.VAL}/loss_epoch',
                                  ylabel="Loss", best_epochs=best_epochs, ax=ax1)
    plot_crossval_training_curves(metrics_df, train_metric=f'{ModelKey.TRAIN}/{MetricsKey.AUROC}',
                                  val_metric=f'{ModelKey.VAL}/{MetricsKey.AUROC}',
                                  ylabel="AUROC", best_epochs=best_epochs, ax=ax2)
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
    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_crossval_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param base_metrics_list: List of metric names to include in the table.
    :param best_epochs: Dictionary mapping each cross-validation index to its best epoch.
    :param metrics_prefix: Prefix to add to the metrics names (e.g. `val`, `test`)
    """
    report.add_heading(heading, level=level)
    metrics_list = [metrics_prefix + metric for metric in base_metrics_list]
    if best_epochs:
        metrics_df = get_best_epoch_metrics(metrics_df, metrics_list, best_epochs)
    metrics_table = get_crossval_metrics_table(metrics_df, metrics_list)
    report.add_tables([metrics_table])


def render_roc_and_pr_curves(report: HTMLReport, heading: str, level: int, report_dir: Path,
                             outputs_dfs: Dict[int, pd.DataFrame], prefix: str = '') -> None:
    """
    Function to render ROC and PR curves for HTML reports.

    :param report: HTML report to perform the rendering.
    :param heading: Heading of the section.
    :param level: Level of HTML heading (e.g. sub-section, sub-sub-section) corresponding to HTML heading levels.
    :param report_dir: Local directory where the report is stored.
    :param outputs_dfs: A dictionary of dataframes with the sorted cross-validation indices as keys.
    :param prefix: Prefix to add to the figures saved (e.g. `val`, `test`).
    """
    report.add_heading(heading, level=level)
    fig = plot_crossval_roc_and_pr_curves(outputs_dfs, scores_column='prob_class1')
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
    :param outputs_dfs: A dictionary of dataframes with the sorted cross-validation indices as keys.
    :param prefix: Prefix to add to the figures saved (e.g. `val`, `test`).
    """
    report.add_heading(heading, level=level)
    fig = plot_confusion_matrices(crossval_dfs=outputs_dfs, class_names=class_names)
    confusion_matrices_fig_path = report_dir / f"{prefix}confusion_matrices.png"
    fig.savefig(confusion_matrices_fig_path, bbox_inches='tight')
    report.add_images([confusion_matrices_fig_path], base64_encode=True)


if __name__ == "__main__":
    """
    Usage example from CLI:
    python generate_crossval_report.py \
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
                         overwrite=args.overwrite)
