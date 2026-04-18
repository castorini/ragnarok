from __future__ import annotations

import json
from typing import Any


def load_prompts_from_file(prompt_file: str) -> dict[str, str]:
    prompts = {}

    try:
        with open(prompt_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    qid = record.get("query", {}).get("qid")
                    prompt = record.get("rag_exec_summary", {}).get("prompt")

                    if qid and prompt:
                        prompts[str(qid)] = prompt

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Error parsing JSON in prompt file on line {line_num}: {e}"
                    )
                    continue
                except Exception as e:
                    print(
                        f"Warning: Error processing prompt record on line {line_num}: {e}"
                    )
                    continue

    except FileNotFoundError:
        print(
            f"Warning: Prompt file '{prompt_file}' not found. Proceeding without external prompts."
        )
    except Exception as e:
        print(f"Warning: Error reading prompt file '{prompt_file}': {e}")

    print(f"Loaded {len(prompts)} prompts from external file.")
    return prompts


def convert_record(
    old_record: dict[str, Any], prompts: dict[str, str] | None = None
) -> dict[str, Any]:
    metadata = {
        "team_id": old_record.get("team_id", "organizer"),
        "run_id": old_record.get("run_id", "unknown-run"),
        "type": old_record.get("type", "automatic"),
        "narrative_id": old_record.get("topic_id", old_record.get("narrative_id", 1)),
        "narrative": old_record.get("topic", old_record.get("narrative", "")),
    }

    topic_id = str(old_record.get("topic_id", old_record.get("narrative_id", "")))
    if prompts and topic_id in prompts:
        metadata["prompt"] = prompts[topic_id]

    references = old_record.get("references", [])
    if not isinstance(references, list):
        references = []

    answer = []
    if "answer" in old_record:
        old_answer = old_record["answer"]
        answer = old_answer

    if not answer:
        answer.append(
            {
                "text": "No answer content available in the original record.",
                "citations": [],
            }
        )

    return {"metadata": metadata, "references": references, "answer": old_answer}


def convert_jsonl_file(
    input_file: str,
    output_file: str,
    prompt_file: str | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    prompts = load_prompts_from_file(prompt_file) if prompt_file else None

    converted_count = 0
    error_count = 0

    with (
        open(input_file, encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                old_record = json.loads(line)
                new_record = convert_record(old_record, prompts)
                json.dump(new_record, outfile, ensure_ascii=False)
                outfile.write("\n")
                converted_count += 1
                if verbose and converted_count % 100 == 0:
                    print(f"Converted {converted_count} records...")
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                error_count += 1
                continue
            except Exception as e:
                if verbose:
                    print(f"Error processing record on line {line_num}: {e}")
                error_count += 1
                continue

    print("Conversion completed!")
    print(f"Successfully converted: {converted_count} records")
    if error_count > 0:
        print(f"Errors encountered: {error_count} records")
    print(f"Output written to: {output_file}")
    return {
        "converted_count": converted_count,
        "error_count": error_count,
        "output_file": output_file,
    }
