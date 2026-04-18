#!/usr/bin/env python3
"""Script to convert JSONL format from old structure to new structure."""

import argparse

from ragnarok.scripts.trec25_conversion import (
    convert_jsonl_file,
)
from ragnarok.scripts.trec25_conversion import (
    convert_record as _convert_record,
)
from ragnarok.scripts.trec25_conversion import (
    load_prompts_from_file as _load_prompts_from_file,
)


def parse_arguments():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Convert JSONL format from old structure to new structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.jsonl converted_data.jsonl
  %(prog)s data.jsonl converted_data.jsonl --prompt-file prompts.jsonl
  %(prog)s input.jsonl output.jsonl --prompt-file prompts.jsonl --verbose
        """,
    )

    parser.add_argument(
        "--input_file", help="Path to the input JSONL file to convert", required=True
    )

    parser.add_argument(
        "--output_file", help="Path to the output JSONL file to create", required=True
    )

    parser.add_argument(
        "--prompt_file",
        "-p",
        dest="prompt_file",
        help="Optional path to a JSONL file containing prompts to include in the output",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress and error information",
    )

    return parser.parse_args()


def main():
    """Main function to handle command line arguments and run conversion."""

    args = parse_arguments()

    print("Converting JSONL format...")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    if args.prompt_file:
        print(f"Prompt file: {args.prompt_file}")
    if args.verbose:
        print("Verbose mode: enabled")
    print("-" * 50)

    convert_jsonl_file(
        input_file=args.input_file,
        output_file=args.output_file,
        prompt_file=args.prompt_file,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()


convert_record = _convert_record
load_prompts_from_file = _load_prompts_from_file
__all__ = ["convert_jsonl_file", "convert_record", "load_prompts_from_file", "main"]
