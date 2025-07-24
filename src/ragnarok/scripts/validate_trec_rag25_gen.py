#!/usr/bin/env python3
import json
import sys
import re
import unicodedata
from argparse import ArgumentParser

MARCODOC = re.compile(r"^msmarco_v2\.1_doc_\d+_\d+#\d+_\d+$")
REQUIRED_METADATA_KEYS = {"team_id", "run_id", "narrative_id"}
ALLOWED_TYPES = {"manual", "automatic"}
RESPONSE_LIMIT = 400
CITATION_LIMIT = 100

def load_topic_ids(path):
    topic_ids = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                topic = json.loads(line)
                topic_ids[str(topic["id"])] = topic["title"]
            except Exception as e:
                print(f"Failed to parse topic line: {line.strip()} ({e})")
    return topic_ids

def compute_response_length(entry):
    total = 0
    for a in entry.get("answer", []):
        text = a.get("text", "").strip()
        tokenized = unicodedata.normalize("NFKC", text)
        total += len(tokenized.split())
    return total

def fix_rag_answer(entry, count, verbose=False):
    current_length = compute_response_length(entry)
    if current_length <= RESPONSE_LIMIT:
        return entry, current_length

    print(f"[Fix-{count}] response_length={current_length} > {RESPONSE_LIMIT}. Trimming answer...")

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

def fix_citations(entry, count, format_type, verbose=False):
    """Remove duplicates from references, trim if they exceed the limit, and update indexes accordingly."""
    refs = entry.get("references", [])
    warnings = []
    
    # Remove duplicates while preserving order
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
    
    # Trim references if they exceed the limit
    if len(refs) > CITATION_LIMIT:
        original_count = len(refs)
        entry["references"] = refs[:CITATION_LIMIT]
        warning_msg = f"references trimmed from {original_count} to {CITATION_LIMIT}"
        warnings.append(warning_msg)
        print(f"[Fix-{count}] WARNING: {warning_msg}")
        
        # Update answer citations for format_type 1 (indexes)
        if format_type == 1:
            for idx, answer in enumerate(entry.get("answer", [])):
                if "citations" in answer:
                    old_citations = answer["citations"]
                    # Filter out citations that are now out of range
                    new_citations = [c for c in old_citations if isinstance(c, int) and 0 <= c < CITATION_LIMIT]
                    
                    if len(new_citations) != len(old_citations):
                        dropped_count = len(old_citations) - len(new_citations)
                        warning_msg = f"answer[{idx}].citations: dropped {dropped_count} out-of-range citation(s)"
                        warnings.append(warning_msg)
                        print(f"[Fix-{count}] WARNING: {warning_msg}")
                        if verbose:
                            dropped_citations = [c for c in old_citations if not (isinstance(c, int) and 0 <= c < CITATION_LIMIT)]
                            print(f"Dropped citations: {dropped_citations}")
                    
                    answer["citations"] = new_citations
        
        # Update answer citations for format_type 2 (segment IDs)
        elif format_type == 2:
            valid_refs = set(entry["references"])
            for idx, answer in enumerate(entry.get("answer", [])):
                if "citations" in answer:
                    old_citations = answer["citations"]
                    # Filter out citations that are no longer in references
                    new_citations = [c for c in old_citations if c in valid_refs]
                    
                    if len(new_citations) != len(old_citations):
                        dropped_count = len(old_citations) - len(new_citations)
                        warning_msg = f"answer[{idx}].citations: dropped {dropped_count} citation(s) not found in trimmed references"
                        warnings.append(warning_msg)
                        print(f"[Fix-{count}] WARNING: {warning_msg}")
                        if verbose:
                            dropped_citations = [c for c in old_citations if c not in valid_refs]
                            print(f"Dropped citations: {dropped_citations}")
                    
                    answer["citations"] = new_citations
    
    return entry, warnings

def validate_entry(entry, format_type, valid_topic_ids):
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
                errors.append(f"metadata.narrative_id '{narrative_id}' not found in topic file")

        if "type" in md and md["type"] not in ALLOWED_TYPES:
            errors.append(f"invalid metadata.type: {md.get('type')}")

        if "type" not in md:
            warnings.append("optional field 'type' is missing from metadata")

        if "narrative" not in md:
            warnings.append("optional field 'narrative' is missing from metadata. Added narrative field.")
            md["narrative"] = valid_topic_ids[narrative_id]

        if "prompt" not in md:
            warnings.append("optional field 'prompt' is missing from metadata")

    refs = entry.get("references")
    if not isinstance(refs, list):
        errors.append("references must be a list")
    else:
        for i, ref in enumerate(refs):
            if not isinstance(ref, str) or not MARCODOC.match(ref):
                warnings.append(f"reference[{i}] not in MARCODOC format: {ref}")

    ans = entry.get("answer", [])
    if not isinstance(ans, list):
        errors.append("answer must be a list")
    else:
        for idx, a in enumerate(ans):
            if not isinstance(a, dict):
                errors.append(f"answer[{idx}] must be object")
                continue
            if "text" not in a or not isinstance(a["text"], str):
                errors.append(f"answer[{idx}].text missing or not string")
            if "citations" not in a:
                errors.append(f"answer[{idx}].citations missing")
            else:
                cits = a["citations"]
                if not isinstance(cits, list):
                    errors.append(f"answer[{idx}].citations not a list")
                elif format_type == 1:
                    if not all(isinstance(c, int) for c in cits):
                        errors.append(f"answer[{idx}].citations must be ints (indexes into references)")
                    else:
                        for c in cits:
                            if c < 0 or c >= len(refs):
                                errors.append(f"answer[{idx}].citations index out of range: {c}")
                elif format_type == 2:
                    if not all(isinstance(c, str) for c in cits):
                        errors.append(f"answer[{idx}].citations must be strings (segment IDs)")
                    else:
                        for c in cits:
                            if c not in refs:
                                errors.append(f"answer[{idx}].citation not found in references: {c}")

    return errors, warnings

def main():
    p = ArgumentParser(description="Validate and optionally fix TREC RAG 2025 AG output format.")
    p.add_argument("--input", help="JSONL input file or '-' for stdin")
    p.add_argument("--format", type=int, choices=[1, 2], default=1, help="Citation format: 1=indexes, 2=segment IDs")
    p.add_argument("--topics", required=True, help="Path to TREC RAG 2025 topic file (JSONL with 'id')")
    p.add_argument("--fix-length", action="store_true", help=f"Trim answers to {RESPONSE_LIMIT} tokens if needed", default=True)
    p.add_argument("--fix-citations", action="store_true", help=f"Trim citations to {CITATION_LIMIT} if needed and update indexes", default=True)
    p.add_argument("--verbose", action="store_true", help="Print details when trimming")
    args = p.parse_args()

    valid_topic_ids = load_topic_ids(args.topics)
    input_stream = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    
    # Store all entries and track if any fixes were made
    entries_to_write = []
    any_fixes_made = False
    
    total_errors = 0
    total_warnings = 0
    for i, line in enumerate(input_stream, 1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            original_entry = json.loads(json.dumps(entry))  # Deep copy for comparison
        except json.JSONDecodeError as e:
            print(f"[Line {i}] ❌ JSON decode error: {e}")
            total_errors += 1
            continue

        # Apply fixes
        fix_warnings = []
        entry_was_fixed = False
        
        if args.fix_citations:
            entry, citation_warnings = fix_citations(entry, i, args.format, verbose=args.verbose)
            fix_warnings.extend(citation_warnings)
            if citation_warnings:
                entry_was_fixed = True

        if args.fix_length:
            original_length = compute_response_length(original_entry)
            entry, current_length = fix_rag_answer(entry, i, verbose=args.verbose)
            if current_length != original_length:
                entry_was_fixed = True
        else:
            current_length = compute_response_length(entry)

        if entry_was_fixed:
            any_fixes_made = True

        errors, warnings = validate_entry(entry, args.format, valid_topic_ids)

        if errors:
            print(f"[Line {i}] ❌ ERRORS:")
            for e in errors:
                print(f"Error: {e}")
            total_errors += 1
        else:
            if args.verbose:
                print(f"[Line {i}] ✅ OK (Length: {current_length} tokens)")

        # Print all warnings (including fix warnings)
        all_warnings = fix_warnings + warnings
        if len(all_warnings) > 0:
            total_warnings += 1
        for w in all_warnings:
            if not w.startswith("answer[") and not w.startswith("references"):  # Don't duplicate fix warnings
                print(f"[Line {i}] WARNING: {w}")

        # Store entry for potential output
        if args.fix_length or args.fix_citations:
            entries_to_write.append(entry)

    # Write output file only if fixes were made
    if any_fixes_made and (args.fix_length or args.fix_citations):
        output_filename = f"{args.input}.fixed" if args.input != "-" else "stdin.fixed"
        print(f"\nFixes were applied. Writing output to: {output_filename}")
        with open(output_filename, "w", encoding="utf-8") as output_stream:
            for entry in entries_to_write:
                output_stream.write(json.dumps(entry) + "\n")
    elif (args.fix_length or args.fix_citations) and not any_fixes_made and args.verbose:
        print("\nNo fixes were needed. Output file not created.")

    if total_errors:
        print(f"\nValidation completed: {total_errors} line(s) with errors.")
        sys.exit(1)
    elif total_warnings:
        print("\nValidation completed: all lines passed (with possible warnings).")
    else:
        print("\nValidation completed: all lines passed.")

if __name__ == "__main__":
    main()
