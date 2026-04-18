import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dacite import from_dict


@dataclass
class Query:
    text: str
    qid: str | int


@dataclass
class Candidate:
    docid: str | int
    score: float
    doc: dict[str, Any]


@dataclass
class Request:
    query: Query
    candidates: list[Candidate] = field(default_factory=list)
    # Optional Ranking Exec Summary
    ranking_exec_summary: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RAGExecInfo:
    prompt: Any
    response: Any
    input_token_count: int
    output_token_count: int
    reasoning: str | None = None
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class CitedSentence:
    text: str
    citations: list[int] = field(default_factory=list)


@dataclass
class Result:
    query: Query
    references: list[str | int] = field(default_factory=list)
    answer: list[CitedSentence] = field(default_factory=list)
    rag_exec_summary: RAGExecInfo | None = None


class OutputFormat(Enum):
    JSON = "json"
    JSONL = "jsonl"

    def __str__(self):
        return self.value


def result_to_dict(
    result: Result,
    run_id: str,
    *,
    include_trace: bool = False,
    redact_prompts: bool = False,
) -> dict[str, Any]:
    record = {
        "run_id": run_id,
        "topic_id": result.query.qid,
        "topic": result.query.text,
        "references": result.references,
        "response_length": sum(
            len(sentence.text.replace(",", " ").replace(";", " ").split())
            for sentence in result.answer
        ),
        "answer": [
            {"text": sentence.text, "citations": sentence.citations}
            for sentence in result.answer
        ],
    }
    reasoning = None
    if result.rag_exec_summary is not None:
        reasoning = result.rag_exec_summary.reasoning
    if reasoning:
        record["reasoning_traces"] = [reasoning]
    if include_trace and result.rag_exec_summary is not None:
        record["trace"] = {
            "prompt": None if redact_prompts else result.rag_exec_summary.prompt,
            "response": result.rag_exec_summary.response,
            "input_token_count": result.rag_exec_summary.input_token_count,
            "output_token_count": result.rag_exec_summary.output_token_count,
        }
    return record


def write_results_jsonl(results: list[Result], filename: str, run_id: str) -> None:
    with Path(filename).open("w", encoding="utf-8") as file_obj:
        for result in results:
            json.dump(result_to_dict(result, run_id), file_obj)
            file_obj.write("\n")


def _load_json_records(file_path: str) -> list[dict[str, Any]]:
    extension = file_path.split(".")[-1]
    with open(file_path, encoding="utf-8") as file_obj:
        if extension == "jsonl":
            return [json.loads(line) for line in file_obj if line.strip()]
        if extension == "json":
            loaded = json.load(file_obj)
            return loaded if isinstance(loaded, list) else [loaded]
    raise ValueError(f"Expected json or jsonl file format, got {extension}")


def _result_from_dict(result_dict: dict[str, Any]) -> Result:
    return Result(
        query=Query(text=result_dict["topic"], qid=result_dict["topic_id"]),
        references=result_dict["references"],
        answer=[
            CitedSentence(text=sentence["text"], citations=sentence["citations"])
            for sentence in result_dict["answer"]
        ],
        rag_exec_summary=None,
    )


def read_requests_from_file(file_path: str) -> list[Request]:
    return [
        from_dict(data_class=Request, data=request_dict)
        for request_dict in _load_json_records(file_path)
    ]


def read_results_from_file(file_path: str) -> list[Result]:
    return [
        _result_from_dict(result_dict) for result_dict in _load_json_records(file_path)
    ]


def remove_unused_references(result: Result, max_per_sentence: int = 3) -> Result:
    # Find all referenced document ids in the citations
    cited_docids = set()
    for cited_sentence in result.answer:
        cited_sentence.citations = cited_sentence.citations[:max_per_sentence]
        cited_docids.update(cited_sentence.citations)

    # Filter the references list to only include cited docids
    result.references = [
        ref for i, ref in enumerate(result.references) if i in cited_docids
    ]

    # Create a mapping from old indices to new indices
    new_index_map = {
        old_idx: new_idx for new_idx, old_idx in enumerate(sorted(cited_docids))
    }

    # Update citations in CitedSentence
    for cited_sentence in result.answer:
        cited_sentence.citations = [
            new_index_map[old_idx] for old_idx in cited_sentence.citations
        ]

    return result


class DataWriter:
    def __init__(
        self,
        data: Request | Result | list[Result] | list[Request],
        append: bool = False,
    ):
        if isinstance(data, list):
            self._data = data
        else:
            self._data = [data]
        self._append = append

    def write_rag_exec_summary(self, filename: str):
        with open(filename, "a" if self._append else "w") as f:
            for d in self._data:
                rag_exec_summary = d.rag_exec_summary.__dict__.copy()
                if rag_exec_summary.get("reasoning") is None:
                    rag_exec_summary.pop("reasoning", None)
                exec_summary = {
                    "query": d.query.__dict__,
                    "rag_exec_summary": rag_exec_summary,
                }
                f.write(json.dumps(exec_summary) + "\n")

    def _convert_result_to_dict(self, result: Result, run_id: str) -> dict[str, Any]:
        return result_to_dict(result, run_id)

    def write_in_json_format(self, filename: str, run_id: str):
        formatted_data = [self._convert_result_to_dict(d, run_id) for d in self._data]
        with open(filename, "a" if self._append else "w") as f:
            json.dump(formatted_data, f, indent=2)

    def write_in_jsonl_format(self, filename: str, run_id: str):
        with open(filename, "a" if self._append else "w") as f:
            for d in self._data:
                json.dump(self._convert_result_to_dict(d, run_id), f)
                f.write("\n")
