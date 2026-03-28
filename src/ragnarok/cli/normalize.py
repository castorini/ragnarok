from __future__ import annotations

from typing import Any, cast


def unwrap_direct_generate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema_version = payload.get("schema_version")
    artifacts = payload.get("artifacts")
    if schema_version != "castorini.cli.v1" or not isinstance(artifacts, list):
        return payload

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_payload = artifact.get("data", artifact.get("value"))
        if (
            isinstance(artifact_payload, dict)
            and {
                "query",
                "candidates",
            }
            <= artifact_payload.keys()
        ):
            return artifact_payload
        if isinstance(artifact_payload, list):
            if len(artifact_payload) != 1:
                raise ValueError(
                    "direct generate envelope input requires exactly one record"
                )
            record = artifact_payload[0]
            if isinstance(record, dict) and {"query", "candidates"} <= record.keys():
                return record

    raise ValueError(
        "direct generate envelope input must contain a single artifact record "
        "with query and candidates"
    )


def _normalize_query(query_payload: Any) -> Any:
    from ragnarok.data import Query

    if isinstance(query_payload, str):
        return Query(text=query_payload, qid="q0")
    if isinstance(query_payload, dict) and isinstance(query_payload.get("text"), str):
        return Query(text=query_payload["text"], qid=query_payload.get("qid", "q0"))
    raise ValueError("query must be a string or an object with a text field")


def _normalize_candidate(candidate_payload: Any, index: int) -> Any:
    from ragnarok.data import Candidate

    if isinstance(candidate_payload, str):
        return Candidate(
            docid=f"d{index}",
            score=0.0,
            doc={"segment": candidate_payload},
        )
    if isinstance(candidate_payload, dict):
        if isinstance(candidate_payload.get("text"), str):
            return Candidate(
                docid=candidate_payload.get("docid", f"d{index}"),
                score=float(candidate_payload.get("score", 0.0)),
                doc={"segment": candidate_payload["text"]},
            )
        doc = candidate_payload.get("doc")
        if isinstance(doc, str):
            return Candidate(
                docid=candidate_payload.get("docid", f"d{index}"),
                score=float(candidate_payload.get("score", 0.0)),
                doc={"segment": doc},
            )
        if isinstance(doc, dict):
            segment = doc.get("segment") or doc.get("contents")
            if isinstance(segment, str):
                return Candidate(
                    docid=candidate_payload.get("docid", f"d{index}"),
                    score=float(candidate_payload.get("score", 0.0)),
                    doc={"segment": segment},
                )
    raise ValueError(
        "each candidate must be a string, {text: ...}, {doc: '...'}, "
        "or {doc: {segment|contents: ...}}"
    )


def normalize_direct_generate_input(payload: dict[str, Any]) -> Any:
    from ragnarok.data import Request

    payload = unwrap_direct_generate_payload(payload)
    candidates_payload = payload.get("candidates")
    if not isinstance(candidates_payload, list):
        raise ValueError("candidates must be a list")
    return Request(
        query=_normalize_query(payload.get("query")),
        candidates=[
            _normalize_candidate(candidate_payload, index)
            for index, candidate_payload in enumerate(candidates_payload)
        ],
        ranking_exec_summary=cast(
            list[dict[str, Any]], payload.get("ranking_exec_summary", [])
        ),
    )
