#!/usr/bin/env python3
import sys
from argparse import ArgumentParser

import ragnarok.scripts.rag25_validation as rag25_validation


def validate_rag25_file(
    input_path: str,
    topics_path: str,
    format_type: int = 1,
    fix_length: bool = False,
    fix_citations: bool = False,
    verbose: bool = False,
):
    return rag25_validation.validate_rag25_entries(
        input_path=input_path,
        topics_path=topics_path,
        format_type=format_type,
        fix_length=fix_length,
        fix_citations_flag=fix_citations,
        verbose=verbose,
    )


def main():
    p = ArgumentParser(
        description="Validate and optionally fix TREC RAG 2025 AG output format."
    )
    p.add_argument("--input", help="JSONL input file or '-' for stdin")
    p.add_argument(
        "--format",
        type=int,
        choices=[1, 2],
        default=1,
        help="Citation format: 1=indexes, 2=segment IDs",
    )
    p.add_argument(
        "--topics",
        required=True,
        help="Path to TREC RAG 2025 topic file (JSONL with 'id')",
    )
    p.add_argument(
        "--fix-length",
        action="store_true",
        help=f"Trim answers to {rag25_validation.RESPONSE_LIMIT} tokens if needed",
    )
    p.add_argument(
        "--fix-citations",
        action="store_true",
        help=f"Trim citations to {rag25_validation.CITATION_LIMIT} if needed and update indexes",
    )
    p.add_argument("--verbose", action="store_true", help="Print details when trimming")
    args = p.parse_args()

    summary = validate_rag25_file(
        input_path=args.input,
        topics_path=args.topics,
        format_type=args.format,
        fix_length=args.fix_length,
        fix_citations=args.fix_citations,
        verbose=args.verbose,
    )
    if summary["fixed_output_path"] is not None:
        print(
            f"\nFixes were applied. Writing output to: {summary['fixed_output_path']}"
        )
    if summary["error_count"]:
        print(f"\nValidation completed: {summary['error_count']} line(s) with errors.")
        sys.exit(1)
    if summary["warning_count"]:
        print("\nValidation completed: all lines passed (with possible warnings).")
    else:
        print("\nValidation completed: all lines passed.")


if __name__ == "__main__":
    main()


load_topic_ids = rag25_validation.load_topic_ids
compute_response_length = rag25_validation.compute_response_length
fix_rag_answer = rag25_validation.fix_rag_answer
fix_citations_fn = rag25_validation.fix_citations
validate_entry = rag25_validation.validate_entry
RESPONSE_LIMIT = rag25_validation.RESPONSE_LIMIT
CITATION_LIMIT = rag25_validation.CITATION_LIMIT
