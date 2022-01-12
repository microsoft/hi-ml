#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import param
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

# Need this as otherwise a description of all the params in a class is added to the class docstring
# which makes generated documentation with sphinx messy.
param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False


@dataclass
class ParserResult:
    """
    Stores the results of running an argument parser, broken down into a argument-to-value dictionary,
    arguments that the parser does not recognize.
    """
    args: Dict[str, Any]
    unknown: List[str]
    overrides: Dict[str, Any]


def _create_default_namespace(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Creates an argparse Namespace with all parser-specific default values set.

    :param parser: The parser to work with.
    :return:
    """
    # This is copy/pasted from parser.parse_known_args
    namespace = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(namespace, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(namespace, action.dest, action.default)
    for dest in parser._defaults:
        if not hasattr(namespace, dest):
            setattr(namespace, dest, parser._defaults[dest])
    return namespace


def parse_arguments(parser: argparse.ArgumentParser,
                    fail_on_unknown_args: bool = False,
                    args: List[str] = None) -> ParserResult:
    """
    Parses a list of commandline arguments with a given parser. Returns results broken down into a full
    arguments dictionary, a dictionary of arguments that were set to non-default values, and unknown
    arguments.

    :param parser: The parser to use
    :param fail_on_unknown_args: If True, raise an exception if the parser encounters an argument that it does
        not recognize. If False, unrecognized arguments will be ignored, and added to the "unknown" field of
        the parser result.
    :param args: Arguments to parse. If not given, use those in sys.argv
    :return: The parsed arguments, and overrides
    """
    if args is None:
        args = sys.argv[1:]
    # The following code is a slightly modified version of what happens in parser.parse_known_args. This had to be
    # copied here because otherwise we would not be able to achieve the priority order that we desire.
    namespace = _create_default_namespace(parser)

    try:
        namespace, unknown = parser._parse_known_args(args, namespace)
        if hasattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR):
            unknown.extend(getattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR))
            delattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR)
    except argparse.ArgumentError:
        parser.print_usage(sys.stderr)
        err = sys.exc_info()[1]
        parser._print_message(str(err), sys.stderr)
        raise
    # Parse the arguments a second time, without supplying defaults, to see which arguments actually differ
    # from defaults.
    namespace_without_defaults, _ = parser._parse_known_args(args, argparse.Namespace())
    parsed_args = vars(namespace).copy()
    overrides = vars(namespace_without_defaults).copy()
    if len(unknown) > 0 and fail_on_unknown_args:
        raise ValueError(f'Unknown arguments: {unknown}')
    return ParserResult(
        args=parsed_args,
        unknown=unknown,
        overrides=overrides,
    )
