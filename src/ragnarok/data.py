import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union

from dacite import from_dict


@dataclass
class Query:
    text: str
    qid: Union[str | int]


@dataclass
class Candidate:
    docid: Union[str | int]
    score: float
    doc: Dict[str, Any]


@dataclass
class Request:
    query: Query
    candidates: List[Candidate] = field(default_factory=list)
    # Optional Ranking Exec Summary
    ranking_exec_summary: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RAGExecInfo:
    prompt: Any
    response: Any
    input_token_count: int
    output_token_count: int
    candidates: List[Candidate] = field(default_factory=list)


@dataclass
class CitedSentence:
    text: str
    citations: List[int] = field(default_factory=list)


@dataclass
class Result:
    query: Query
    references: List[Union[str | int]] = field(default_factory=list)
    answer: List[CitedSentence] = field(default_factory=list)
    rag_exec_summary: RAGExecInfo = None


class OutputFormat(Enum):
    JSON = "json"
    JSONL = "jsonl"

    def __str__(self):
        return self.value


def read_requests_from_file(file_path: str) -> List[Request]:
    extension = file_path.split(".")[-1]
    if extension == "jsonl":
        requests = []
        with open(file_path, "r") as f:
            for l in f:
                if not l.strip():
                    continue
                requests.append(from_dict(data_class=Request, data=json.loads(l)))
        return requests
    elif extension == "json":
        with open(file_path, "r") as f:
            request_dicts = json.load(f)
        return [
            from_dict(data_class=Request, data=request_dict)
            for request_dict in request_dicts
        ]
    else:
        raise ValueError(f"Expected json or jsonl file format, got {extension}")

def read_results_from_file(file_path: str) -> List[Result]:
    extension = file_path.split(".")[-1]
    if extension == "jsonl":
        results = []
        with open(file_path, "r") as f:
            for l in f:
                if not l.strip():
                    continue
                result_dict = json.loads(l)
                result = Result(
                    query=Query(
                        text=result_dict["topic"], 
                        qid=result_dict["topic_id"]
                    ),
                    references=result_dict["references"],
                    answer=[
                        CitedSentence(
                            text=sentence["text"], 
                            citations=sentence["citations"]
                        )
                        for sentence in result_dict["answer"]
                    ],
                    rag_exec_summary=None
                )
                results.append(result)
        return results
    elif extension == "json":
        with open(file_path, "r") as f:
            result_dicts = json.load(f)
        return [
            Result(
                query=Query(
                    text=result_dict["topic"], 
                    qid=result_dict["topic_id"]
                ),
                references=result_dict["references"],
                answer=[
                    CitedSentence(
                        text=sentence["text"], 
                        citations=sentence["citations"]
                    )
                    for sentence in result_dict["answer"]
                ],
                rag_exec_summary=None
            )
            for result_dict in result_dicts
        ]
    else:
        raise ValueError(f"Expected json or jsonl file format, got {extension}")

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
        data: Union[Request | Result | List[Result] | List[Request]],
        append: bool = False,
    ):
        if isinstance(data, list):
            self._data = data
        else:
            self._data = [data]
        self._append = append

    def write_rag_exec_summary(self, filename: str):
        exec_summary = []
        with open(filename, "a" if self._append else "w") as f:
            for d in self._data:
                exec_summary = {
                    "query": d.query.__dict__,
                    "rag_exec_summary": d.rag_exec_summary.__dict__,
                }
                f.write(json.dumps(exec_summary) + "\n")

    def _convert_result_to_dict(self, result: Result) -> Dict:
        result_dict = {
            "topic_id": result.query.qid,
            "topic": result.query.text,
            "references": result.references,
            "response_length": sum(len(sentence.text) for sentence in result.answer),
            "answer": [
                {"text": sentence.text, "citations": sentence.citations}
                for sentence in result.answer
            ],
        }
        return result_dict

    def write_in_json_format(self, filename: str):
        formatted_data = [self._convert_result_to_dict(d) for d in self._data]
        with open(filename, "a" if self._append else "w") as f:
            json.dump(formatted_data, f, indent=2)

    def write_in_jsonl_format(self, filename: str):
        with open(filename, "a" if self._append else "w") as f:
            for d in self._data:
                json.dump(self._convert_result_to_dict(d), f)
                f.write("\n")
