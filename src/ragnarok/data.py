import json
from dataclasses import dataclass, field
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
    rag_exec_summary: List[RAGExecInfo] = (field(default_factory=list),)


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
        for d in self._data:
            values = []
            for info in d.rag_exec_summary:
                values.append(info.__dict__)
            exec_summary.append(
                {"query": d.query.__dict__, "rag_exec_summary": values}
            )
        with open(filename, "a" if self._append else "w") as f:
            json.dump(exec_summary, f, indent=2)

    def _convert_result_to_dict(self, result: Result) -> Dict:
        result_dict = {
            "topic_id": result.query.qid,
            "topic": result.query.text,
            "references": result.references,
            "response_length": sum(len(sentence.text) for sentence in result.answer),
            "answer": [{"text": sentence.text, "citations": sentence.citations} for sentence in result.answer]
        }
        return result_dict

    def write_in_json_format(self, filename: str):
        formatted_data = [self._convert_result_to_dict(d) for d in self._data]
        with open(filename, 'a' if self._append else 'w') as f:
            json.dump(formatted_data, f, indent=2)

    def write_in_jsonl_format(self, filename: str):
        with open(filename, 'a' if self._append else 'w') as f:
            for d in self._data:
                json.dump(self._convert_result_to_dict(d), f)
                f.write('\n')