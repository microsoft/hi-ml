#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from azureml.core import Workspace
from matplotlib import pyplot as plt

from health_azure.utils import get_aml_run_from_run_id, get_workspace
from health_ml.utils.reports import HTMLReport
from histopathology.utils.analysis_plot_utils import (add_training_curves_legend, plot_confusion_matrices, plot_crossval_roc_and_pr_curves,
                                                      plot_crossval_training_curves, plot_confusion_matrices)
from histopathology.utils.output_utils import (AML_LEGACY_TEST_OUTPUTS_CSV, AML_TEST_OUTPUTS_CSV,
                                               AML_VAL_OUTPUTS_CSV)
from histopathology.utils.report_utils import (collect_crossval_metrics, collect_crossval_outputs,
                                               crossval_runs_have_val_and_test_outputs, get_best_epoch_metrics,
                                               get_best_epochs, get_crossval_metrics_table, get_formatted_run_info)
from histopathology.utils.naming import MetricsKey


def generate_html_report(parent_run_id: str, output_dir: Path,
                         class_names: str,
                         workspace_config_path: Optional[Path] = None,
                         include_test: bool = False, overwrite: bool = False) -> None:
    aml_workspace = get_workspace(workspace_config_path=workspace_config_path)
    parent_run = get_aml_run_from_run_id(parent_run_id, aml_workspace=aml_workspace)
    report_dir = output_dir / parent_run.display_name
    report_dir.mkdir(parents=True, exist_ok=True)

    report = HTMLReport(output_folder=report_dir)

    report.add_text(get_formatted_run_info(parent_run))

    report.add_heading("Azure ML metrics", level=2)

    # Download metrics from AML. Can take several seconds for each child run
    metrics_df = collect_crossval_metrics(parent_run_id, report_dir, aml_workspace, overwrite=overwrite)
    best_epochs = get_best_epochs(metrics_df, 'val/auroc', maximise=True)

    # Add training curves for loss and AUROC (train and val.)
    render_training_curves(report, heading="Training curves", level=3,
                           metrics_df=metrics_df, best_epochs=best_epochs, report_dir=report_dir)

    # Add tables with relevant metrics (val. and test)
    class_names = class_names.split(sep=",")
    n_classes = len(class_names)
    if n_classes == 2:
        base_metrics_list = [MetricsKey.ACC, MetricsKey.AUROC, MetricsKey.PRECISION, MetricsKey.RECALL, MetricsKey.F1]
    elif n_classes > 1:
        base_metrics_list = [MetricsKey.ACC, MetricsKey.AUROC]
    else:
        raise ValueError(f"Invalid number of classes {n_classes} inferred from {class_names}, these should be > = 2.")
    base_metrics_list += class_names

    render_metrics_table(report, heading="Validation metrics (best epoch)", level=3,
                         metrics_df=metrics_df, best_epochs=best_epochs,
                         base_metrics_list=base_metrics_list, metrics_prefix='val/')

    if include_test:
        render_metrics_table(report, heading="Test metrics", level=3,
                             metrics_df=metrics_df, best_epochs=None,
                             base_metrics_list=base_metrics_list, metrics_prefix='test/')

    report.add_heading("Model outputs", level=2)

    has_val_and_test_outputs = crossval_runs_have_val_and_test_outputs(parent_run)

    if n_classes == 2:
        # Currently ROC and PR curves rendered only for binary case
        # TODO: Enable rendering of multi-class ROC and PR curves

        if has_val_and_test_outputs:
            # Add val. ROC and PR curves
            render_roc_and_pr_curves(report, "Validation ROC and PR curves", level=3,
                                    parent_run_id=parent_run_id, aml_workspace=aml_workspace, report_dir=report_dir,
                                    output_filename=AML_VAL_OUTPUTS_CSV, overwrite=overwrite, prefix='val_')

        if include_test:
            # Add test ROC and PR curves
            test_outputs_filename = AML_TEST_OUTPUTS_CSV if has_val_and_test_outputs else AML_LEGACY_TEST_OUTPUTS_CSV
            render_roc_and_pr_curves(report, "Test ROC and PR curves", level=3,
                                    parent_run_id=parent_run_id, aml_workspace=aml_workspace, report_dir=report_dir,
                                    output_filename=test_outputs_filename, overwrite=overwrite, prefix='test_')

    if has_val_and_test_outputs:
        # Add val. confusion matrices
        render_confusion_matrices(report, "Validation Confusion Matrices", level=3, class_names=class_names,
                                parent_run_id=parent_run_id, aml_workspace=aml_workspace, report_dir=report_dir,
                                output_filename=AML_VAL_OUTPUTS_CSV, overwrite=overwrite, prefix='val_')
    
    if include_test:
        # Add test confusion matrices
        render_confusion_matrices(report, "Test Confusion Matrices", level=3, class_names=class_names,
                                parent_run_id=parent_run_id, aml_workspace=aml_workspace, report_dir=report_dir,
                                output_filename=test_outputs_filename, overwrite=overwrite, prefix='test_')
        
    print(f"Rendering report to: {report.report_path_html.absolute()}")
    report.render()


def render_training_curves(report: HTMLReport, heading: str, level: int,
                           metrics_df: pd.DataFrame, best_epochs: Optional[Dict[int, int]],
                           report_dir: Path) -> None:

    report.add_heading(heading, level=level)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plot_crossval_training_curves(metrics_df, train_metric='train/loss_epoch', val_metric='val/loss_epoch',
                                  ylabel="Loss", best_epochs=best_epochs, ax=ax1)
    plot_crossval_training_curves(metrics_df, train_metric='train/auroc', val_metric='val/auroc',
                                  ylabel="AUROC", best_epochs=best_epochs, ax=ax2)
    add_training_curves_legend(fig, include_best_epoch=True)
    training_curves_fig_path = report_dir / "training_curves.png"
    fig.savefig(training_curves_fig_path, bbox_inches='tight')
    report.add_images([training_curves_fig_path], base64_encode=True)


def render_metrics_table(report: HTMLReport, heading: str, level: int,
                         metrics_df: pd.DataFrame, best_epochs: Optional[Dict[int, int]],
                         base_metrics_list: List[str], metrics_prefix: str) -> None:

    report.add_heading(heading, level=level)
    metrics_list = [metrics_prefix + metric for metric in base_metrics_list]
    if best_epochs:
        metrics_df = get_best_epoch_metrics(metrics_df, metrics_list, best_epochs)
    metrics_table = get_crossval_metrics_table(metrics_df, metrics_list)
    report.add_tables([metrics_table])


def render_roc_and_pr_curves(report: HTMLReport, heading: str, level: int,
                             parent_run_id: str, aml_workspace: Workspace, report_dir: Path,
                             output_filename: str, overwrite: bool, prefix: str = '') -> None:

    report.add_heading(heading, level=level)
    outputs_dfs = collect_crossval_outputs(parent_run_id, report_dir, aml_workspace, output_filename=output_filename,
                                           overwrite=overwrite)
    fig = plot_crossval_roc_and_pr_curves(outputs_dfs, scores_column='prob_class1')
    roc_pr_curves_fig_path = report_dir / f"{prefix}roc_pr_curves.png"
    fig.savefig(roc_pr_curves_fig_path, bbox_inches='tight')
    report.add_images([roc_pr_curves_fig_path], base64_encode=True)


def render_confusion_matrices(report: HTMLReport, heading: str, level: int, class_names: List[str],
                             parent_run_id: str, aml_workspace: Workspace, report_dir: Path,
                             output_filename: str, overwrite: bool, prefix: str = '') -> None:

    report.add_heading(heading, level=level)
    outputs_dfs = collect_crossval_outputs(parent_run_id, report_dir, aml_workspace, output_filename=output_filename,
                                           overwrite=overwrite)
    fig = plot_confusion_matrices(crossval_dfs=outputs_dfs, class_names=class_names)
    confusion_matrices_fig_path = report_dir / f"{prefix}confusion_matrices.png"
    fig.savefig(confusion_matrices_fig_path, bbox_inches='tight')
    report.add_images([confusion_matrices_fig_path], base64_encode=True)


if __name__ == "__main__":
    """
    Usage example from CLI:
    generate_crossval_report.py \
    --run_id "HD_4ab0d833-fe55-44e8-aa04-cbaadbcc2733" \
    --output_dir "outputs" \
    --class_names "ISUP 0,ISUP 1,ISUP 2,ISUP 3,ISUP 4,ISUP 5" \
    -- include_test
    """

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--run_id', help="The parent Hyperdrive run ID.")
    parser.add_argument('--output_dir', help="Directory where to download Azure ML data and save the report.")
    parser.add_argument('--class_names', help="Names of classes separated by comma, same as used in the Hyperdrive run. "
                                               "These are used to find the per-class metrics from Azure ML.")
    parser.add_argument('--workspace_config', help="Path to Azure ML workspace config.json file. "
                                                   "If omitted, will try to load default workspace.")
    parser.add_argument('--include_test', action='store_true', help="Opt-in flag to include test results "
                                                                    "in the generated report.")
    parser.add_argument('--overwrite', action='store_true', help="Forces (re)download of metrics and output files, "
                                                                 "even if they already exist locally.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path.cwd() / "outputs"
    workspace_config = Path(args.workspace_config).absolute() if args.workspace_config else None

    print(f"Output dir: {Path(args.output_dir).absolute()}")
    if workspace_config is not None:
        if not workspace_config.is_file():
            raise ValueError(f"Specified workspace config file does not exist: {workspace_config}")
        print(f"Workspace config: {workspace_config}")

    generate_html_report(parent_run_id=args.run_id,
                         output_dir=Path(args.output_dir),
                         class_names=args.class_names,
                         workspace_config_path=workspace_config,
                         include_test=args.include_test,
                         overwrite=args.overwrite)
