# Ragnarok Input/Output Examples

## Generate Input (batch JSONL)

Each line in the input file:

```json
{
  "query": {"qid": "q1", "text": "What is information retrieval?"},
  "candidates": [
    {"docid": "d1", "score": 10.0, "doc": {"segment": "Information retrieval is the science of searching for information..."}},
    {"docid": "d2", "score": 8.0, "doc": {"segment": "IR systems help users find relevant documents..."}}
  ]
}
```

Lightweight shorthand:

```json
{
  "query": "What is information retrieval?",
  "candidates": ["Information retrieval is the science of searching...", "IR systems help users find relevant documents..."]
}
```

## Generate Output (JSONL)

```json
{
  "run_id": "ragnarok",
  "topic_id": "q1",
  "topic": "What is information retrieval?",
  "references": ["d1", "d2"],
  "response_length": 45,
  "answer": [
    {"text": "Information retrieval (IR) is the science of searching for information in documents and databases.", "citations": [0]},
    {"text": "IR systems help users find relevant documents from large collections.", "citations": [0, 1]},
    {"text": "Modern IR leverages techniques from natural language processing and machine learning.", "citations": [1]}
  ]
}
```

With `--include-trace`:

```json
{
  "run_id": "ragnarok",
  "topic_id": "q1",
  "topic": "What is information retrieval?",
  "references": ["d1", "d2"],
  "response_length": 45,
  "answer": [...],
  "trace": {
    "prompt": "...",
    "response": "...",
    "input_token_count": 512,
    "output_token_count": 150
  }
}
```

## Answer Structure

Each answer is an array of `CitedSentence` objects:

```json
{
  "text": "Sentence text here.",
  "citations": [0, 1]
}
```

- `citations` are zero-indexed into the `references` array
- `references` maps indices to `docid` values from input candidates

## TREC RAG 2025 Submission Format

After `ragnarok convert trec25-format`:

```json
{
  "team_id": "castorini",
  "run_id": "ragnarok",
  "narrative_id": "q1",
  "type": "automatic",
  "references": ["msmarco_v2.1_doc_00_0#0_0", "msmarco_v2.1_doc_01_0#1_0"],
  "response_length": 45,
  "answer": [
    {"text": "...", "citations": [0, 1]}
  ]
}
```

Constraints:
- Response limit: 400 words
- Citation limit: 100 references
- Document IDs must match MS MARCO v2.1 format: `msmarco_v2.1_doc_\d+_\d+#\d+_\d+`
