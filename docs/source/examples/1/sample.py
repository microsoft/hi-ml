#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from argparse import ArgumentParser
from typing import List


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
    parser = ArgumentParser()
    parser.add_argument("-n", "--count", type=int, default=100,
                        required=False, help="Maximum value (not included)")
    args = parser.parse_args()

    primes = sieve(args.count)
    print("\n".join(map(str, primes)))


if __name__ == "__main__":
    main()
