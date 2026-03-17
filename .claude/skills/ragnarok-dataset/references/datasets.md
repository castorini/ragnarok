# Ragnarok Datasets and Retrieval Methods

## Known Dataset Names

Dataset names follow the pattern `<track>.<split>`:

| Dataset | Track | Description |
|---------|-------|-------------|
| `rag24.raggy-dev` | TREC RAG 2024 | RAGgy dev split |
| `rag24.researchy-dev` | TREC RAG 2024 | Researchy dev split |

Check available datasets via:
```bash
ragnarok describe generate --output json
```

## Retrieval Methods

| Method | Type | Stage | Notes |
|--------|------|-------|-------|
| `bm25` | Sparse | First-stage | Fast, no GPU needed, requires pyserini index |
| `rank_zephyr` | Neural | Reranking | Zephyr-based listwise reranker |
| `rank_zephyr_rho` | Neural | Reranking | Zephyr-Rho variant (often better than base) |
| `rank_vicuna` | Neural | Reranking | Vicuna-based listwise reranker |
| `gpt-4o` | LLM | Reranking | GPT-4o listwise reranking (requires API key) |
| `gpt-4` | LLM | Reranking | GPT-4 listwise reranking |
| `gpt-3.5-turbo` | LLM | Reranking | GPT-3.5 listwise reranking |

## Common Multi-Stage Configurations

```bash
# BM25 → Zephyr-Rho → Generation (recommended)
--retrieval-method bm25,rank_zephyr_rho --topk 100,20

# BM25 → GPT-4o reranking → Generation (higher quality, higher cost)
--retrieval-method bm25,gpt-4o --topk 100,20

# BM25 only → Generation (fastest, lowest quality)
--retrieval-method bm25 --topk 20

# Three-stage: BM25 → Zephyr → GPT-4o → Generation
--retrieval-method bm25,rank_zephyr,gpt-4o --topk 100,50,10
```

## topk Best Practices

| Use Case | topk | Rationale |
|----------|------|-----------|
| Fast prototype | `20` | Minimal candidates, fast turnaround |
| Standard evaluation | `100,20` | Good recall/precision balance |
| High quality | `1000,100,20` | Maximum recall, three-stage pipeline |
| Context-limited model | `100,5` | Few candidates for small context windows |

## Pyserini Requirements

```bash
# Install
uv sync --extra pyserini

# Verify
python3 -c "import pyserini; print(pyserini.__version__)"

# Check Java
java -version  # Needs JDK 21+
```

Indexes are cached locally. First-time dataset access downloads the index.
