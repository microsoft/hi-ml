from pathlib import Path

from matplotlib import pyplot as plt

from health_azure.utils import get_workspace
from health_ml.utils.reports import HTMLReport
from histopathology.utils.report_utils import (add_training_curves_legend, collect_crossval_metrics,
                                               collect_crossval_outputs, get_best_epoch_metrics, get_best_epochs,
                                               get_crossval_metrics_table, get_formatted_run_info,
                                               plot_crossval_roc_and_pr_curves, plot_crossval_training_curves)


def generate_html_report(parent_run_id: str, download_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aml_workspace = get_workspace()

    report = HTMLReport(output_folder=output_dir)

    report.add_text(get_formatted_run_info(parent_run_id, aml_workspace))

    report.add_heading("Azure ML metrics", level=2)

    # Download metrics from AML. Can take several seconds for each child run
    metrics_df = collect_crossval_metrics(parent_run_id, download_dir, aml_workspace)
    best_epochs = get_best_epochs(metrics_df, 'val/auroc', maximise=True)

    # Add training curves for loss and AUROC (train and val.)
    report.add_heading("Training curves", level=3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plot_crossval_training_curves(metrics_df, train_metric='train/loss_epoch', val_metric='val/loss_epoch',
                                  ylabel="Loss", best_epochs=best_epochs, ax=ax1)
    plot_crossval_training_curves(metrics_df, train_metric='train/auroc', val_metric='val/auroc',
                                  ylabel="AUROC", best_epochs=best_epochs, ax=ax2)
    add_training_curves_legend(fig, include_best_epoch=True)
    training_curves_fig_path = output_dir / "training_curves.png"
    fig.savefig(training_curves_fig_path, bbox_inches='tight')
    report.add_images([training_curves_fig_path], base64_encode=True)

    # Add tables with relevant metrics (val. and test)
    base_metrics_list = ['accuracy', 'auroc', 'f1score', 'precision', 'recall', '0', '1']

    report.add_heading("Validation metrics (best epoch)", level=3)
    val_metrics_list = ['val/' + metric for metric in base_metrics_list]
    val_metrics_df = get_best_epoch_metrics(metrics_df, val_metrics_list, best_epochs)
    val_metrics_table = get_crossval_metrics_table(val_metrics_df, val_metrics_list)
    report.add_tables([val_metrics_table])

    report.add_heading("Test metrics", level=3)
    test_metrics_list = ['test/' + metric for metric in base_metrics_list]
    test_metrics_table = get_crossval_metrics_table(metrics_df, test_metrics_list)
    report.add_tables([test_metrics_table])

    # Add test ROC and PR curves
    num_crossval_splits = len(metrics_df.columns)
    crossval_dfs = collect_crossval_outputs(parent_run_id, download_dir, num_crossval_splits)

    report.add_heading("Test ROC and PR curves", level=2)
    fig = plot_crossval_roc_and_pr_curves(crossval_dfs)
    cohort_roc_pr_curves_fig_path = output_dir / "roc_pr_curves.png"
    fig.savefig(cohort_roc_pr_curves_fig_path, bbox_inches='tight')
    report.add_images([cohort_roc_pr_curves_fig_path], base64_encode=True)

    print(f"Rendering report to: {report.report_path_html.absolute()}")
    report.render()


if __name__ == "__main__":
    from argparse import ArgumentParser
    output_dir = Path(__file__).absolute().parent.parent.parent.parent.parent / "outputs"
    print(f"Output dir: {output_dir}")

    parser = ArgumentParser()
    parser.add_argument('--run_id', help="The parent Hyperdrive run ID")
    parser.add_argument('--download_dir', help="The parent Hyperdrive run ID")
    parser.add_argument('--output_dir', help="The ")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path.cwd() / "outputs"
    if args.download_dir is None:
        args.download_dir = args.output_dir

    generate_html_report(parent_run_id=args.run_id,
                         download_dir=Path(args.download_dir),
                         output_dir=Path(args.output_dir))
