# Ragnarök

[![PyPI](https://img.shields.io/pypi/v/pyragnarok?color=brightgreen)](https://pypi.org/project/pyragnarok/)
[![Downloads](https://static.pepy.tech/personalized-badge/pyragnarok?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/pyragnarok)
[![Downloads](https://static.pepy.tech/personalized-badge/pyragnarok?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/week)](https://pepy.tech/project/pyragnarok)
<!-- [![Generic badge](https://img.shields.io/badge/arXiv-2309.15088-red.svg)](https://arxiv.org/abs/2309.15088) -->
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)


Ragnarök is a battleground for the best retrieval-augmented generation (RAG) models!


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
`uv run python src/ragnarok/scripts/run_ragnarok.py --help` or
`uv run python examples/rag_demo.py --help`.

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
- `ragnarok validate`: validate request payloads or TREC output artifacts
- `ragnarok convert trec25-format`: convert older generation outputs into the
  newer TREC 2025 format
- `ragnarok describe`: inspect command metadata and examples
- `ragnarok schema`: print supported JSON schemas
- `ragnarok doctor`: report environment and dependency readiness

### Direct And Introspection Examples

```bash
ragnarok generate \
  --model-path gpt-4o \
  --input-json '{"query":"how long is life cycle of flea","candidates":["The life cycle of a flea can last anywhere from 20 days to an entire year."]}' \
  --prompt-mode chatqa \
  --output json
```

```bash
ragnarok describe generate --output json
ragnarok schema generate-direct-input --output json
ragnarok doctor --output json
```


## RAG

We have a wide range of models supported by Ragnarök.
To run the `command-r-plus` model on the `rag24.researchy-dev` topics using the top-20 `bm25` results from the MS MARCO v2.1 segment collection, you can run the following command:
```bash
ragnarok generate --model-path command-r-plus --topk 20 \
  --dataset rag24.researchy-dev --retrieval-method bm25 --prompt-mode cohere \
  --context-size 8192 --max-output-tokens 1024
```

Or to run the `gpt-4o` model (ChatQA inspired format) on the `rag24.raggy-dev` topics with multi-stage retrieval + reranking ()`bm25` followed by `rank_zephyr_rho`) and augmented-generation on the top-5 MS MARCO v2.1 segments, you can run the following command:
```bash
ragnarok generate --model-path gpt-4o --topk 100,5 \
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

For a small inline-hit RAG smoke test without preparing a dataset-backed
retrieval run, use:

```bash
uv run python examples/rag_demo.py --model gpt-4o
```

Pass `--use_azure_openai` for Azure OpenAI, `--include_reasoning` to capture
reasoning where supported, and `--print_prompt` when you want to inspect the
rendered prompt.

## Contributing 

If you would like to contribute to the project, please refer to the [contribution guidelines](CONTRIBUTING.md).

## 🦙🐧 Model Zoo

Most LLMs supported by VLLM/FastChat should additionally be supported by Ragnarök too, albeit we do not test all of them. If you would like to see a specific model added, please open an issue or a pull request. The following is a table of generation models which we regularly use with Ragnarök:

| Model Name        | Identifier/Link                            |
|-------------------|---------------------------------------------|
| GPT-4o            | `gpt-4o`                                   |
| GPT-4           | `gpt-4`                              |
| GPT-3.5-turbo    | `gpt-35-turbo`                            |
| command-r-plus    | `command-r-plus`                     |
| command-r         | `command-r`                          |
| Llama-3 8B Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Llama3-ChatQA-1.5 | `nvidia/Llama3-ChatQA-1.5` |


## ✨ References

If you use Ragnarök, please cite the following:

[[2406.16828] Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track](https://arxiv.org/abs/2406.16828)

<!-- {% raw %} -->
```
@ARTICLE{pradeep2024ragnarok,
  title   = {{Ragnarök}: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track},
  author  = {Ronak Pradeep and Nandan Thakur and Sahel Sharifymoghaddam and Eric Zhang and Ryan Nguyen and Daniel Campos and Nick Craswell and Jimmy Lin},
  year    = {2024},
  journal = {arXiv:2406.16828},
}
```

## 🙏 Acknowledgments

This research is supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.
