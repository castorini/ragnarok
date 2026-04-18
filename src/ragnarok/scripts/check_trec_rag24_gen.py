#!/usr/bin/env python3

import sys
import traceback

from ragnarok.scripts.rag24_validation import check_rag24_run

# Check a RAG track generation output for
# - missing or non-matching run_id
# - incorrect or missing topics
# - malformed references
# - no sentences
# - generation length > 400


class Errlog:
    """This is meant to be used in a context manager, for example
    with Errlog(foo) as log:
        ...
    If not, be sure to call .close() when done.
    """

    def __init__(self, runfile, max_errors=930):
        self.filename = runfile + ".errlog"
        print(f"Writing errors to {self.filename}")
        self.fp = open(self.filename, "w")
        self.error_count = 0
        self.max_errors = max_errors

    def __enter__(self):
        return self

    def close(self):
        if self.error_count == 0:
            print("No errors", file=self.fp)
        self.fp.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def error(self, line, msg):
        print(f"ERROR Line {line}: {msg}", file=self.fp)
        self.error_count += 1
        if self.error_count > self.max_errors:
            raise Exception(f"{line} Stopping, too many errors")

    def warn(self, line, msg):
        print(f"WARNING Line {line}: {msg}", file=self.fp)


def check_rag_gen_run(args, log):
    check_rag24_run(topicfile=args.topicfile, runfile=args.runfile, log=log)


def run_check_rag24_output(topicfile: str, runfile: str) -> dict:
    class Args:
        def __init__(self, topicfile: str, runfile: str):
            self.topicfile = topicfile
            self.runfile = runfile

    args = Args(topicfile=topicfile, runfile=runfile)
    with Errlog(args.runfile) as log:
        try:
            check_rag_gen_run(args, log)
        except Exception as exc:
            log.error(-1, exc)
            traceback.print_exc()
            return {"valid": False, "error_count": log.error_count, "runfile": runfile}
        return {
            "valid": log.error_count == 0,
            "error_count": log.error_count,
            "errlog_path": runfile + ".errlog",
            "fixed_run_path": runfile + ".fixed",
            "runfile": runfile,
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Checker for normal TREC runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "-f", "--topicfile", required=True, help="File containing topic IDs"
    )
    ap.add_argument("runfile")

    args = ap.parse_args()

    summary = run_check_rag24_output(args.topicfile, args.runfile)
    sys.exit(0 if summary["valid"] else 255)
