from __future__ import annotations

import collections
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Protocol

MARCODOC = re.compile(r"^msmarco_v2\.1_doc_\d+_\d+#\d+_\d+$")


class Rag24Log(Protocol):
    error_count: int

    def error(self, line: int, msg: str) -> None: ...
    def warn(self, line: int, msg: str) -> None: ...


def load_topics(topicfile: str) -> tuple[collections.Counter[str], dict[str, str]]:
    topics: collections.Counter[str] = collections.Counter()
    queries: dict[str, str] = {}
    topic_path = Path(topicfile)
    if not topic_path.exists():
        topic_path = Path(sys.path[0]) / topicfile
        if not topic_path.exists():
            raise FileNotFoundError(f"{topicfile} not found")

    with open(topic_path, encoding="utf-8") as fp:
        print("Reading topics from", topic_path)
        for line in fp:
            topic_id, query = line.strip().split("\t")
            topics[topic_id] = 0
            queries[topic_id] = query
    return topics, queries


def compute_rag24_response_length(answer: list[dict[str, Any]]) -> int:
    length = 0
    for sent in answer:
        text = str(sent["text"]).strip()
        tokenized = unicodedata.normalize("NFKC", text)
        length += len(tokenized.split())
    return length


def fix_rag24_answer(
    obj: dict[str, Any],
    current_length: int,
    *,
    count: int,
    log: Rag24Log,
) -> tuple[dict[str, Any], int]:
    log.warn(count, f"Attempting to fix RAG answer of length {current_length}")
    answer = obj["answer"]
    while current_length > 400 and answer:
        last_sentence = answer.pop()
        text = last_sentence["text"].strip()
        tokenized = unicodedata.normalize("NFKC", text)
        tokens = tokenized.split()
        current_length -= len(tokens)
        log.warn(count, f"Removing a sentence from the end: {text}")
        log.warn(count, f"Updated length: {current_length}")

    obj["response_length"] = current_length
    return obj, current_length


def check_rag24_run(
    *,
    topicfile: str,
    runfile: str,
    log: Rag24Log,
) -> str:
    the_runtag = None
    topics, queries = load_topics(topicfile)
    write_fixed_file = runfile + ".fixed"
    with (
        open(runfile, encoding="utf-8") as run,
        open(write_fixed_file, "w", encoding="utf-8") as fixed_run,
    ):
        print("Reading run from", runfile)
        count = 0
        for line in run:
            count += 1
            length = 0

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as j:
                log.error(count, f"Error parsing JSON line at {j.colno}")
                continue

            for field in (
                "run_id",
                "topic_id",
                "topic",
                "references",
                "response_length",
                "answer",
            ):
                if field not in obj:
                    log.error(count, f'Entry is missing "{field}" field.')
                    break
            else:
                run_id = obj["run_id"]
                if the_runtag is None:
                    the_runtag = run_id
                elif the_runtag != obj["run_id"]:
                    log.error(
                        count,
                        f'Run tag inconsistent ("{obj["run_id"]}" instead of "{the_runtag}")',
                    )

                topic_id = obj["topic_id"]
                if topic_id not in topics:
                    log.error(count, f"Unknown topic ({topic_id})")
                    continue

                if obj["topic"] != queries[topic_id]:
                    log.error(
                        count,
                        "Topic text does not match official topic for this topic ID",
                    )

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

                if obj["response_length"] > 400:
                    log.warn(count, "Reported response_length is too long")

                for sent in obj["answer"]:
                    text = sent["text"].strip()
                    tokenized = unicodedata.normalize("NFKC", text)
                    tokens = tokenized.split()
                    length += len(tokens)

                    if len(sent["citations"]) >= 1 and (
                        max(sent["citations"]) + 1 > num_refs
                        or min(sent["citations"]) < 0
                    ):
                        log.warn(
                            count,
                            "Response sentence has a citation that is out of bounds",
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
                    obj, length = fix_rag24_answer(obj, length, count=count, log=log)

                if length != obj["response_length"]:
                    log.warn(
                        count,
                        f"Reported RAG answer ({obj['response_length']}) is not equal to actual response length ({length}), maybe you did not NFCK normalize the text or strip characters?",
                    )
                    obj["response_length"] = length
                topics[topic_id] += 1
                fixed_run.write(json.dumps(obj) + "\n")
                continue

        print(f"Wrote fixed run to {write_fixed_file}")
        for topic in topics:
            if topics[topic] == 0:
                log.warn(count, f"No response returned for topic {topic}")
            elif topics[topic] > 1:
                log.error(
                    count,
                    f"Too many responses ({topics[topic]}) generated for topic {topic}",
                )
    return write_fixed_file
