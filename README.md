# Ragnarök

[![PyPI](https://img.shields.io/pypi/v/pyragnarok?color=brightgreen)](https://pypi.org/project/pyragnarok/)
[![Downloads](https://static.pepy.tech/personalized-badge/pyragnarok?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/pyragnarok)
[![Downloads](https://static.pepy.tech/personalized-badge/pyragnarok?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/week)](https://pepy.tech/project/pyragnarok)
<!-- [![Generic badge](https://img.shields.io/badge/arXiv-2309.15088-red.svg)](https://arxiv.org/abs/2309.15088) -->
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)


Ragnarök is a battleground for the best retrieval-augmented generation (RAG) models!

## Releases

- Current version: `0.0.1`
- Release notes: [docs/release-notes/release-notes-v0.0.1.md](docs/release-notes/release-notes-v0.0.1.md)


## 📟 Instructions

### Source Installation

`uv` is the canonical contributor workflow for this repository. The existing
conda path remains available for contributors who want it.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

For development from source:

```bash
git clone https://github.com/castorini/ragnarok.git
cd ragnarok
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev
```

If you prefer not to activate the virtual environment, use `uv run`, for example
`uv run ragnarok --help`, `uv run pre-commit run --all-files`,
`uv run ragnarok-quality-gate`, or `uv run python examples/rag_demo.py --help`.

Install optional stacks only when you need them:

```bash
uv sync --group dev --extra cloud
uv sync --group dev --extra local
uv sync --group dev --extra api
uv sync --group dev --extra pyserini
uv sync --group dev --extra all
```

If you want to keep using conda, create a Python 3.11 environment and install
the same base package dependencies:

```bash
conda create -n ragnarok python=3.11 -y
conda activate ragnarok
pip install -r requirements.txt
pip install -e .
```

Then install any optional stack you need, for example `pip install -e ".[cloud]"`
or `pip install -e ".[local]"`.

### PyPI Installation

```bash
pip install pyragnarok
```

## CLI

`ragnarok ...` is now the canonical offline command-line interface for this
repository. Prefer it over calling `src/ragnarok/scripts/*.py` directly. In an
activated environment, run `ragnarok ...`; otherwise use `uv run ragnarok ...`.

### Command Overview

- `ragnarok generate`: run dataset-backed generation, batch request-file
  generation, or direct single-request generation
- `ragnarok serve`: start a FastAPI server for direct single-request generation
- `ragnarok validate`: validate request payloads or TREC output artifacts
- `ragnarok convert trec25-format`: convert older generation outputs into the
  newer TREC 2025 format
- `ragnarok describe`: inspect command metadata and examples
- `ragnarok schema`: print supported JSON schemas
- `ragnarok doctor`: report environment and dependency readiness
- `ragnarok view`: inspect an existing generation artifact without re-running a model

### Direct And Introspection Examples

```bash
ragnarok generate \
  --model gpt-4o \
  --input-json '{"query":"how long is life cycle of flea","candidates":["The life cycle of a flea can last anywhere from 20 days to an entire year."]}' \
  --prompt-mode chatqa \
  --output json
```

To opt into async generation for direct JSON or request-file generation, add
`--execution-mode async`. You can also tune request fan-out with
`--max-concurrency`, for example:

```bash
ragnarok generate \
  --model gpt-4o \
  --input-file requests.jsonl \
  --output-file results.jsonl \
  --prompt-mode chatqa \
  --execution-mode async \
  --max-concurrency 8
```

```bash
ragnarok describe generate --output json
ragnarok schema generate-direct-input --output json
ragnarok validate generate --input-json '{"query":"q","candidates":["p"]}' --output json
ragnarok doctor --output json
ragnarok view results.jsonl --records 1
```

Serve the direct generation API:

```bash
ragnarok serve \
  --model gpt-4o \
  --prompt-mode chatqa \
  --port 8083

curl -X POST http://127.0.0.1:8083/v1/generate \
  -H 'content-type: application/json' \
  -d '{"query":"q","candidates":["p"]}'

curl -s "http://127.0.0.1:8081/v1/msmarco-v1-passage/search?query=what%20is%20python%20commonly%20used%20for" \
  | curl -s -X POST http://127.0.0.1:8083/v1/generate \
      -H 'content-type: application/json' \
      --data-binary @- \
  | jq
```

For TREC RAG 2025 output validation, `ragnarok validate rag25-output ...` is
non-mutating by default. If you explicitly want repairable issues written to a
`.fixed` artifact, add `--apply-fixes` or one of the fix flags.


## RAG

We have a wide range of models supported by Ragnarök.
To run the `command-r-plus` model on the `rag24.researchy-dev` topics using the top-20 `bm25` results from the MS MARCO v2.1 segment collection, you can run the following command:
```bash
ragnarok generate --model command-r-plus --topk 20 \
  --dataset rag24.researchy-dev --retrieval-method bm25 --prompt-mode cohere \
  --context-size 8192 --max-output-tokens 1024
```

Or to run the `gpt-4o` model (ChatQA inspired format) on the `rag24.raggy-dev` topics with multi-stage retrieval + reranking (`bm25` followed by `rank_zephyr_rho`) and augmented-generation on the top-5 MS MARCO v2.1 segments, you can run the following command:
```bash
ragnarok generate --model gpt-4o --topk 100,5 \
    --dataset rag24.raggy-dev --retrieval-method bm25,rank_zephyr_rho --prompt-mode chatqa \
    --context-size 8192 --max-output-tokens 1024 --use-azure-openai
```

If you want Ragnarok to persist model reasoning in the execution-summary sidecar
written under `rag_execution_summary/`, add `--include-reasoning`. This is
currently intended for OpenAI-compatible responses that expose reasoning fields
and open-weight models that emit `<think>...</think>` blocks. The public TREC
result file under `results/` is unchanged.

For OpenAI-compatible models that support effort controls, you can also pass
`--reasoning-effort none|minimal|low|medium|high|xhigh`. Ragnarok forwards that
setting only on the OpenAI-compatible generation path.

### Quick Demo

For the default async inline-hit RAG smoke test without preparing a
dataset-backed retrieval run, use:

```bash
uv run python examples/rag_demo.py --model gpt-4o
```

Pass `--use_azure_openai` for Azure OpenAI, `--include_reasoning` to capture
reasoning where supported, `--max_concurrency` to control async request fan-out,
and `--print_prompt` when you want to inspect the rendered prompt.

If you want the synchronous compatibility demo instead, run:

```bash
uv run python examples/sync_rag_demo.py --model gpt-4o
```

For an opt-in live smoke test that exercises the packaged CLI against a real
OpenAI-compatible backend, run:

```bash
RAGNAROK_LIVE_OPENAI_SMOKE=1 uv run pytest -q -m live test
```

## Testing Tiers

Ragnarök keeps regression coverage in three layers:

- `core`: fast deterministic unit and CLI tests that always run in PR CI
- `integration`: deterministic offline CLI regressions backed by frozen fixtures
- `live`: provider-backed smoke tests gated behind explicit environment variables

Typical local commands:

```bash
uv run ragnarok-quality-gate
uv run pytest -q -m core test
uv run pytest -q -m integration test
RAGNAROK_LIVE_OPENAI_SMOKE=1 uv run pytest -q -m live test
```

## Contributing 

If you would like to contribute to the project, please refer to the [contribution guidelines](CONTRIBUTING.md).

## 🦙🐧 Model Zoo

Ragnarok does not require a hardcoded model whitelist for most common cloud and
open-weight generation setups. In practice, most models exposed through
OpenAI-compatible APIs, OpenRouter, and vLLM can be used as long as they are
compatible with the selected backend and prompt path.

Instead of maintaining a static list of model identifiers in this README, use
the upstream model catalogs:

- OpenAI models: [platform.openai.com/docs/models](https://platform.openai.com/docs/models)
- OpenRouter models: [openrouter.ai/models](https://openrouter.ai/models)
- vLLM supported models: [docs.vllm.ai/en/latest/models/supported_models.html](https://docs.vllm.ai/en/latest/models/supported_models.html)

If you find a backend or model family that should work but does not, open an
issue or pull request with the exact model identifier, backend, and failure
mode.


## ✨ References

If you use Ragnarök, please cite the following:

Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track. Proceedings of the 47th European Conference on Information Retrieval (ECIR 2025), Part I.

<!-- {% raw %} -->
```
@INPROCEEDINGS{pradeep2025ragnarok,
  author    = {Ronak Pradeep and Nandan Thakur and Sahel Sharifymoghaddam and Eric Zhang and Ryan Nguyen and Daniel Campos and Nick Craswell and Jimmy Lin},
  title     = {{Ragnarök}: A Reusable {RAG} Framework and Baselines for {TREC} 2024 {Retrieval-Augmented} {Generation} {Track}},
  booktitle = {Proceedings of the 47th European Conference on Information Retrieval (ECIR 2025), Part I},
  pages     = {132--148},
  year      = {2025},
  address_  = {Lucca, Italy}
}
```

## 🙏 Acknowledgments

This research is supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.
