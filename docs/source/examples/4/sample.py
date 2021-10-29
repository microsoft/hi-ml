#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from health_azure import submit_to_azure_if_needed


def sieve(n: int) -> List[int]:
    """
    A simple implementation of the http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes

    :param n: Maximum value to search up to, not included.
    :return: List of primes upto but not including n.
    """
    all_numbers = [True] * n

    for i in range(2, int(n ** 0.5 + 1)):
        if all_numbers[i]:
            for f in range(i * i, n, i):
                all_numbers[f] = False
    primes = []
    for i in range(2, n):
        if all_numbers[i]:
            primes.append(i)
    return primes


def main() -> None:
    run_info = submit_to_azure_if_needed(
        compute_cluster_name="lite-testing-ds2",
        default_datastore="himldatasets",
        output_datasets=["himl_sample4_output"],
        wait_for_completion=True,
        wait_for_completion_show_output=True)

    parser = ArgumentParser()
    parser.add_argument("-n", "--count", type=int, default=100, required=False, help="Maximum value (not included)")
    parser.add_argument("-o", "--output", type=str, default="primes.txt", required=False, help="Output file name")
    args = parser.parse_args()

    primes = sieve(args.count)
    output_folder = run_info.output_datasets[0] or Path("outputs") / "himl_sample4_output"
    output_folder.mkdir(parents=True, exist_ok=True)
    output = output_folder / args.output
    output.write_text("\n".join(map(str, primes)))


if __name__ == "__main__":
    main()
