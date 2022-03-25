#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt

from health_azure.utils import get_aml_run_from_run_id, get_workspace
from health_ml.utils.reports import HTMLReport
from histopathology.utils.analysis_plot_utils import (add_training_curves_legend, plot_crossval_roc_and_pr_curves,
                                                      plot_crossval_training_curves)
from histopathology.utils.report_utils import (collect_crossval_metrics, collect_crossval_outputs,
                                               crossval_runs_have_val_and_test_outputs, get_best_epoch_metrics,
                                               get_best_epochs, get_crossval_metrics_table, get_formatted_run_info)


def generate_html_report(parent_run_id: str, output_dir: Path, workspace_config_path: Optional[Path] = None,
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
    report.add_heading("Training curves", level=3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plot_crossval_training_curves(metrics_df, train_metric='train/loss_epoch', val_metric='val/loss_epoch',
                                  ylabel="Loss", best_epochs=best_epochs, ax=ax1)
    plot_crossval_training_curves(metrics_df, train_metric='train/auroc', val_metric='val/auroc',
                                  ylabel="AUROC", best_epochs=best_epochs, ax=ax2)
    add_training_curves_legend(fig, include_best_epoch=True)
    training_curves_fig_path = report_dir / "training_curves.png"
    fig.savefig(training_curves_fig_path, bbox_inches='tight')
    report.add_images([training_curves_fig_path], base64_encode=True)

    # Add tables with relevant metrics (val. and test)
    base_metrics_list = ['accuracy', 'auroc', 'f1score', 'precision', 'recall', '0', '1']

    report.add_heading("Validation metrics (best epoch)", level=3)
    val_metrics_list = ['val/' + metric for metric in base_metrics_list]
    val_metrics_df = get_best_epoch_metrics(metrics_df, val_metrics_list, best_epochs)
    val_metrics_table = get_crossval_metrics_table(val_metrics_df, val_metrics_list)
    report.add_tables([val_metrics_table])

    if include_test:
        report.add_heading("Test metrics", level=3)
        test_metrics_list = ['test/' + metric for metric in base_metrics_list]
        test_metrics_table = get_crossval_metrics_table(metrics_df, test_metrics_list)
        report.add_tables([test_metrics_table])

    report.add_heading("Model outputs", level=2)

    has_val_and_test_outputs = crossval_runs_have_val_and_test_outputs(parent_run)

    if has_val_and_test_outputs:
        # Add val. ROC and PR curves
        val_outputs_filename = "val/test_output.csv"
        val_outputs_dfs = collect_crossval_outputs(parent_run_id, report_dir, aml_workspace,
                                                   output_filename=val_outputs_filename,
                                                   overwrite=overwrite)

        report.add_heading("Validation ROC and PR curves", level=3)
        fig = plot_crossval_roc_and_pr_curves(val_outputs_dfs, scores_column='prob_class1')
        val_roc_pr_curves_fig_path = report_dir / "val_roc_pr_curves.png"
        fig.savefig(val_roc_pr_curves_fig_path, bbox_inches='tight')
        report.add_images([val_roc_pr_curves_fig_path], base64_encode=True)

    if include_test:
        # Add test ROC and PR curves
        test_outputs_filename = "test_output.csv"
        if has_val_and_test_outputs:
            test_outputs_filename = "test/" + test_outputs_filename
        test_outputs_dfs = collect_crossval_outputs(parent_run_id, report_dir, aml_workspace,
                                                    output_filename=test_outputs_filename,
                                                    overwrite=overwrite)

        report.add_heading("Test ROC and PR curves", level=3)
        fig = plot_crossval_roc_and_pr_curves(test_outputs_dfs, scores_column='prob_class1')
        test_roc_pr_curves_fig_path = report_dir / "test_roc_pr_curves.png"
        fig.savefig(test_roc_pr_curves_fig_path, bbox_inches='tight')
        report.add_images([test_roc_pr_curves_fig_path], base64_encode=True)

    print(f"Rendering report to: {report.report_path_html.absolute()}")
    report.render()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--run_id', help="The parent Hyperdrive run ID")
    parser.add_argument('--output_dir', help="Directory where to download Azure ML data and save the report")
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

    print(f"Output dir: {args.output_dir.absolute()}")
    if workspace_config is not None:
        if not workspace_config.is_file():
            raise ValueError(f"Specified workspace config file does not exist: {workspace_config}")
        print(f"Workspace config: {workspace_config}")

    generate_html_report(parent_run_id=args.run_id,
                         output_dir=Path(args.output_dir),
                         workspace_config_path=workspace_config,
                         include_test=args.include_test,
                         overwrite=args.overwrite)
