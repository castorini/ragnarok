from __future__ import annotations

import json
import re
import sys
import unicodedata
from typing import Any

MARCODOC = re.compile(r"^msmarco_v2\.1_doc_\d+_\d+#\d+_\d+$")
REQUIRED_METADATA_KEYS = {"team_id", "run_id", "narrative_id"}
ALLOWED_TYPES = {"manual", "automatic"}
RESPONSE_LIMIT = 400
CITATION_LIMIT = 100


def load_topic_ids(path: str) -> dict[str, str]:
    topic_ids = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                topic = json.loads(line)
                topic_ids[str(topic["id"])] = topic["title"]
            except Exception as e:
                print(f"Failed to parse topic line: {line.strip()} ({e})")
    return topic_ids


def compute_response_length(entry: dict[str, Any]) -> int:
    total = 0
    answers = entry.get("answer", [])
    if not isinstance(answers, list):
        return total
    for answer in answers:
        text = answer.get("text", "").strip()
        tokenized = unicodedata.normalize("NFKC", text)
        total += len(tokenized.split())
    return total


def fix_rag_answer(
    entry: dict[str, Any], count: int, verbose: bool = False
) -> tuple[dict[str, Any], int]:
    current_length = compute_response_length(entry)
    if current_length <= RESPONSE_LIMIT:
        return entry, current_length

    print(
        f"[Fix-{count}] response_length={current_length} > {RESPONSE_LIMIT}. Trimming answer..."
    )

    answer = entry["answer"]
    while current_length > RESPONSE_LIMIT and answer:
        last = answer.pop()
        text = last["text"].strip()
        tokenized = unicodedata.normalize("NFKC", text)
        length = len(tokenized.split())
        current_length -= length
        if verbose:
            print(f"Removed: {text} ({length} tokens)")
    return entry, current_length


def fix_citations(
    entry: dict[str, Any], count: int, format_type: int, verbose: bool = False
) -> tuple[dict[str, Any], list[str]]:
    refs = entry.get("references", [])
    warnings = []
    if not isinstance(refs, list):
        refs = []

    original_count = len(refs)
    seen = set()
    unique_refs = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)

    if len(unique_refs) != original_count:
        duplicate_count = original_count - len(unique_refs)
        warning_msg = f"removed {duplicate_count} duplicate reference(s)"
        warnings.append(warning_msg)
        print(f"[Fix-{count}] WARNING: {warning_msg}")
        entry["references"] = unique_refs
        refs = unique_refs

    if len(refs) > CITATION_LIMIT:
        original_count = len(refs)
        entry["references"] = refs[:CITATION_LIMIT]
        warning_msg = f"references trimmed from {original_count} to {CITATION_LIMIT}"
        warnings.append(warning_msg)
        print(f"[Fix-{count}] WARNING: {warning_msg}")

        if format_type == 1:
            answers = entry.get("answer", [])
            if not isinstance(answers, list):
                answers = []
            for idx, answer in enumerate(answers):
                if "citations" not in answer:
                    continue
                old_citations = answer["citations"]
                new_citations = [
                    citation
                    for citation in old_citations
                    if isinstance(citation, int) and 0 <= citation < CITATION_LIMIT
                ]
                if len(new_citations) != len(old_citations):
                    dropped_count = len(old_citations) - len(new_citations)
                    warning_msg = f"answer[{idx}].citations: dropped {dropped_count} out-of-range citation(s)"
                    warnings.append(warning_msg)
                    print(f"[Fix-{count}] WARNING: {warning_msg}")
                    if verbose:
                        dropped_citations = [
                            citation
                            for citation in old_citations
                            if not (
                                isinstance(citation, int)
                                and 0 <= citation < CITATION_LIMIT
                            )
                        ]
                        print(f"Dropped citations: {dropped_citations}")
                answer["citations"] = new_citations
        elif format_type == 2:
            valid_refs = set(entry["references"])
            answers = entry.get("answer", [])
            if not isinstance(answers, list):
                answers = []
            for idx, answer in enumerate(answers):
                if "citations" not in answer:
                    continue
                old_citations = answer["citations"]
                new_citations = [
                    citation for citation in old_citations if citation in valid_refs
                ]
                if len(new_citations) != len(old_citations):
                    dropped_count = len(old_citations) - len(new_citations)
                    warning_msg = f"answer[{idx}].citations: dropped {dropped_count} citation(s) not found in trimmed references"
                    warnings.append(warning_msg)
                    print(f"[Fix-{count}] WARNING: {warning_msg}")
                    if verbose:
                        dropped_citations = [
                            citation
                            for citation in old_citations
                            if citation not in valid_refs
                        ]
                        print(f"Dropped citations: {dropped_citations}")
                answer["citations"] = new_citations

    return entry, warnings


def validate_entry(
    entry: dict[str, Any], format_type: int, valid_topic_ids: dict[str, str]
) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    md = entry.get("metadata")
    if not isinstance(md, dict):
        errors.append("metadata must be an object")
    else:
        missing = REQUIRED_METADATA_KEYS - md.keys()
        if missing:
            errors.append(f"metadata missing keys: {missing}")
        else:
            narrative_id = str(md.get("narrative_id", ""))
            if narrative_id not in valid_topic_ids:
                errors.append(
                    f"metadata.narrative_id '{narrative_id}' not found in topic file"
                )

        if "type" in md and md["type"] not in ALLOWED_TYPES:
            errors.append(f"invalid metadata.type: {md.get('type')}")

        if "type" not in md:
            warnings.append("optional field 'type' is missing from metadata")

        if "narrative" not in md:
            warnings.append(
                "optional field 'narrative' is missing from metadata. Added narrative field."
            )
            md["narrative"] = valid_topic_ids.get(str(md.get("narrative_id", "")), "")

        if "prompt" not in md:
            warnings.append("optional field 'prompt' is missing from metadata")

    refs = entry.get("references")
    if not isinstance(refs, list):
        errors.append("references must be a list")
        refs = []
    else:
        for i, ref in enumerate(refs):
            if not isinstance(ref, str) or not MARCODOC.match(ref):
                warnings.append(f"reference[{i}] not in MARCODOC format: {ref}")

    ans = entry.get("answer", [])
    if not isinstance(ans, list):
        errors.append("answer must be a list")
    else:
        for idx, answer in enumerate(ans):
            if not isinstance(answer, dict):
                errors.append(f"answer[{idx}] must be object")
                continue
            if "text" not in answer or not isinstance(answer["text"], str):
                errors.append(f"answer[{idx}].text missing or not string")
            if "citations" not in answer:
                errors.append(f"answer[{idx}].citations missing")
                continue
            cits = answer["citations"]
            if not isinstance(cits, list):
                errors.append(f"answer[{idx}].citations not a list")
            elif format_type == 1:
                if not all(isinstance(citation, int) for citation in cits):
                    errors.append(
                        f"answer[{idx}].citations must be ints (indexes into references)"
                    )
                else:
                    for citation in cits:
                        if citation < 0 or citation >= len(refs):
                            errors.append(
                                f"answer[{idx}].citations index out of range: {citation}"
                            )
            elif format_type == 2:
                if not all(isinstance(citation, str) for citation in cits):
                    errors.append(
                        f"answer[{idx}].citations must be strings (segment IDs)"
                    )
                else:
                    for citation in cits:
                        if citation not in refs:
                            errors.append(
                                f"answer[{idx}].citation not found in references: {citation}"
                            )

    return errors, warnings


def validate_rag25_entries(
    *,
    input_path: str,
    topics_path: str,
    format_type: int = 1,
    fix_length: bool = False,
    fix_citations_flag: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    valid_topic_ids = load_topic_ids(topics_path)
    input_stream = (
        sys.stdin if input_path == "-" else open(input_path, encoding="utf-8")
    )

    entries_to_write = []
    any_fixes_made = False
    total_errors = 0
    total_warnings = 0
    fixed_output_path = None
    try:
        for i, line in enumerate(input_stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                original_entry = json.loads(json.dumps(entry))
            except json.JSONDecodeError as e:
                print(f"[Line {i}] JSON decode error: {e}")
                total_errors += 1
                continue

            fix_warnings = []
            entry_was_fixed = False
            if fix_citations_flag:
                entry, citation_warnings = fix_citations(
                    entry, i, format_type, verbose=verbose
                )
                fix_warnings.extend(citation_warnings)
                if citation_warnings:
                    entry_was_fixed = True

            if fix_length:
                original_length = compute_response_length(original_entry)
                entry, current_length = fix_rag_answer(entry, i, verbose=verbose)
                if current_length != original_length:
                    entry_was_fixed = True
            if entry_was_fixed:
                any_fixes_made = True

            errors, warnings = validate_entry(entry, format_type, valid_topic_ids)
            if errors:
                for error in errors:
                    print(f"[Line {i}] Error: {error}")
                total_errors += 1

            all_warnings = fix_warnings + warnings
            if all_warnings:
                total_warnings += 1
            for warning in all_warnings:
                print(f"[Line {i}] WARNING: {warning}")

            if fix_length or fix_citations_flag:
                entries_to_write.append(entry)

        if any_fixes_made and (fix_length or fix_citations_flag):
            fixed_output_path = (
                f"{input_path}.fixed" if input_path != "-" else "stdin.fixed"
            )
            with open(fixed_output_path, "w", encoding="utf-8") as output_stream:
                for entry in entries_to_write:
                    output_stream.write(json.dumps(entry) + "\n")
    finally:
        if input_stream is not sys.stdin:
            input_stream.close()

    return {
        "valid": total_errors == 0,
        "error_count": total_errors,
        "warning_count": total_warnings,
        "fixed_output_path": fixed_output_path,
        "fixes_applied": any_fixes_made,
        "format_type": format_type,
    }
