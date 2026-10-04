"""Command line interface of the program."""
from pathlib import Path
import argparse
from argparse import Namespace


def arguments() -> Namespace:
    """Parse the command line arguments.

    Returns:
        A namespace with the paths ``functions_definition``, ``input``
        and ``output``, each falling back to its default location.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        '--functions_definition', type=Path,
        default=Path('data/input/functions_definition.json'),
        help='JSON file with the available functions')
    parser.add_argument(
        '--input', type=Path,
        default=Path('data/input/function_calling_tests.json'),
        help='JSON file with the prompts to process')
    parser.add_argument(
        '--output', type=Path,
        default=Path('data/output/function_calls.json'),
        help='JSON file to write the result to')
    return parser.parse_args()
