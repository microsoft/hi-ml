import pickle
from pathlib import Path

from matplotlib import pyplot as plt

from health_azure.utils import aggregate_hyperdrive_metrics, get_workspace
from health_ml.utils.reports import HTMLReport
from histopathology.utils.report_utils import (collect_crossval_outputs, get_best_epoch_metrics,
                                               get_crossval_metrics_table, get_formatted_run_info,
                                               plot_crossval_roc_and_pr_curves, plot_crossval_training_curves)


def generate_html_report(parent_run_id: str, download_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aml_workspace = get_workspace()

    report = HTMLReport(output_folder=output_dir)

    report.add_text(get_formatted_run_info(parent_run_id, aml_workspace))

    report.add_heading("Azure ML metrics", level=2)

    metrics_pickle = download_dir / parent_run_id / "aml_metrics.pickle"
    if metrics_pickle.is_file():
        print(f"AML metrics file already exists at {metrics_pickle}")
        with open(metrics_pickle, 'rb') as f:
            metrics_df = pickle.load(f)
    else:
        metrics_df = aggregate_hyperdrive_metrics(run_id=parent_run_id,
                                                  child_run_arg_name="cross_validation_split_index",
                                                  aml_workspace=aml_workspace)
        metrics_pickle.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing AML metrics file to {metrics_pickle}")
        with open(metrics_pickle, 'wb') as f:
            pickle.dump(metrics_df, f)

    base_metrics_list = ['accuracy', 'auroc', 'f1score', 'precision', 'recall', '0', '1']

    fig, ax = plt.subplots()
    plot_crossval_training_curves(metrics_df, 'val/auroc', ax)
    training_auroc_curves_fig_path = output_dir / "training_auroc_curves.png"
    fig.savefig(training_auroc_curves_fig_path, bbox_inches='tight')
    report.add_images([training_auroc_curves_fig_path], base64_encode=True)

    report.add_heading("Validation metrics", level=3)
    val_metrics_list = ['val/' + metric for metric in base_metrics_list]
    val_metrics_df = get_best_epoch_metrics(metrics_df, 'val/auroc', val_metrics_list)
    val_metrics_table = get_crossval_metrics_table(val_metrics_df, val_metrics_list)
    report.add_tables([val_metrics_table])

    report.add_heading("Test metrics", level=3)
    test_metrics_list = ['test/' + metric for metric in base_metrics_list]
    test_metrics_table = get_crossval_metrics_table(metrics_df, test_metrics_list)
    report.add_tables([test_metrics_table])

    num_crossval_splits = len(metrics_df.columns)
    # num_crossval_splits = 5
    crossval_dfs = collect_crossval_outputs(parent_run_id, download_dir, num_crossval_splits)

    report.add_heading("ROC and PR curves", level=2)
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

    generate_html_report(parent_run_id=args.run_id,#"HD_f5ee9174-8c44-4e29-9c49-262a3b53fa80",
                         download_dir=Path(args.download_dir),
                         output_dir=Path(args.output_dir))
