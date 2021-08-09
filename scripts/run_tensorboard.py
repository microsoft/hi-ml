#!/usr/bin/env python3
import os

from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--logpath",
        default="outputs",
        type=str,
        required=False,
        help="Path to Tensorboard log directory"
    )

    args = parser.parse_args()
    log_path = args.logpath
    os.system('tensorboard --logdir=' + log_path)


if __name__ == "__main__":
    main()
