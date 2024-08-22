#!/usr/bin/env python3

import collections
import json
import re
import sys
import traceback
import unicodedata
from pathlib import Path

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
    MARCODOC = re.compile(r"^msmarco_v2\.1_doc_\d+_\d+#\d+_\d+$")
    TOPICNO = re.compile(r"^2024-\d+$")

    def fix_rag_answer(obj, current_length, count):
        log.warn(count, f"Attempting to fix RAG answer of length {current_length}")
        while current_length > 400 and obj["answer"]:
            last_sentence = obj["answer"].pop()
            text = last_sentence["text"].strip()
            tokenized = unicodedata.normalize("NFKC", text)
            tokens = tokenized.split()
            current_length -= len(tokens)
            log.warn(count, f"Removing a sentence from the end: {text}")
            log.warn(count, f"Updated length: {current_length}")

        obj["response_length"] = current_length
        return obj, current_length

    the_runtag = None

    topics = collections.Counter()
    queries = {}

    if args.topicfile:
        topicfile = Path(args.topicfile)
        if not topicfile.exists():
            topicfile = Path(sys.path[0]) / args.topicfile
            if not topicfile.exists():
                raise FileNotFoundError(f"{args.topicfile} not found")

        with open(topicfile, "r") as fp:
            print("Reading topics from", topicfile)
            for line in fp:
                t, q = line.strip().split("\t")
                topics[t] = 0
                queries[t] = q
    write_fixed_file = args.runfile + ".fixed"
    with open(args.runfile, "r") as run, open(write_fixed_file, "w") as fixed_run:
        print("Reading run from", args.runfile)
        count = 0
        for line in run:
            count += 1
            length = 0

            # Check JSON parse
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as j:
                log.error(count, f"Error parsing JSON line at {j.colno}")
                continue

            # Check that all fields are present
            if not "run_id" in obj:
                log.error(count, 'Entry is missing "run_id" field.')
                continue
            if not "topic_id" in obj:
                log.error(count, 'Entry is missing "topic_id" field.')
                continue

            if not "topic" in obj:
                log.error(count, 'Entry is missing "topic" field.')
                continue
            if not "references" in obj:
                log.error(count, 'Entry is missing "references" field.')
                continue
            if not "response_length" in obj:
                log.error(count, 'Entry is missing "response_length" field.')
                continue
            if not "answer" in obj:
                log.error(count, 'Entry is missing "answer" field.')
                continue

            # Check runtag
            run_id = obj["run_id"]
            if the_runtag is None:
                the_runtag = run_id
            elif the_runtag != obj["run_id"]:
                log.error(
                    count,
                    f'Run tag inconsistent ("{obj["run_id"]}" instead of "{the_runtag}")',
                )

            # Check that topic is valid
            topic_id = obj["topic_id"]
            if topic_id not in topics:
                log.error(count, f"Unknown topic ({topic_id})")
                # end checks for this line
                continue

            # Check the topic entry
            if obj["topic"] != queries[topic_id]:
                log.error(
                    count, "Topic text does not match official topic for this topic ID"
                )

            # Check references
            refs = set()
            for ref in obj["references"]:
                if not MARCODOC.match(ref):
                    log.error(count, f"Invalid reference docno {ref}")
                elif ref in refs:
                    log.error(count, f"Duplicate document {ref} in references")
                else:
                    refs.add(ref)

            num_refs = len(obj["references"])
            if num_refs > 20:
                log.error(count, "Too many references (max 20)")

            # Check response length
            if obj["response_length"] > 400:
                log.warn(count, f"Reported response_length is too long")

            # Check answer sentences
            length = 0
            for sent in obj["answer"]:
                text = sent["text"].strip()
                tokenized = unicodedata.normalize("NFKC", text)
                tokens = tokenized.split()
                length += len(tokens)

                if len(sent["citations"]) >= 1 and (
                    max(sent["citations"]) + 1 > num_refs or min(sent["citations"]) < 0
                ):
                    log.warn(
                        count, "Response sentence has a citation that is out of bounds"
                    )
                if len(sent["citations"]) != len(set(sent["citations"])):
                    log.warn(count, "Response sentence has duplicate citations")
                    sent["citations"] = list(set(sent["citations"]))
                cites = set()
                for cite in sent["citations"]:
                    if cite < 0 or cite >= num_refs:
                        log.error(
                            count, f"Response sentence has invalid citation {cite}"
                        )
                    elif cite in cites:
                        log.warn(count, f"Duplicate citation {cite}")
                    else:
                        cites.add(cite)
                sent["citations"] = list(cites)

            if length > 400:
                log.warn(count, f"RAG answer is too long ({length} words)")
                (
                    update_obj,
                    length,
                ) = fix_rag_answer(obj, length, count)
                obj = update_obj

            if length != obj["response_length"]:
                log.warn(
                    count,
                    f'Reported RAG answer ({obj["response_length"]}) is not equal to actual response length ({length}), maybe you did not NFCK normalize the text or strip characters?',
                )
                obj["response_length"] = length
            topics[topic_id] += 1
            fixed_run.write(json.dumps(obj) + "\n")
    print(f"Wrote fixed run to {write_fixed_file}")
    for topic in topics:
        if topics[topic] == 0:
            log.warn(count, f"No response returned for topic {topic}")
        elif topics[topic] > 1:
            log.error(
                count,
                f"Too many responses ({topics[topic]}) generated for topic {topic}",
            )


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

    with Errlog(args.runfile) as log:
        try:
            result = check_rag_gen_run(args, log)
        except Exception as e:
            log.error(-1, e)
            traceback.print_exc()
            sys.exit(255)

        if log.error_count > 0:
            sys.exit(255)
        else:
            sys.exit(0)
