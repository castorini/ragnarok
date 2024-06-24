# Ragnarök

[![PyPI](https://img.shields.io/pypi/v/pyragnarok?color=brightgreen)](https://pypi.org/project/pyragnarok/)
[![Downloads](https://static.pepy.tech/personalized-badge/pyragnarok?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/pyragnarok)
[![Downloads](https://static.pepy.tech/personalized-badge/pyragnarok?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/week)](https://pepy.tech/project/pyragnarok)
<!-- [![Generic badge](https://img.shields.io/badge/arXiv-2309.15088-red.svg)](https://arxiv.org/abs/2309.15088) -->
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)


Ragnarök is a battleground for the best retrieval-augmented generation (RAG) models!


## 📟 Instructions

### Source Installation

Create a new conda environment and install the dependencies:

```bash
conda create -n ragnarok python=3.10 -y
conda activate ragnarok
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # For CUDA 11.8
pip install -r requirements.txt
```

### PyPI Installation

```bash
pip install pyragnarok
```


## Let's generate!

For example, to run the `command-r-plus` model on the `rag24.researchy-dev_large` dataset with `bm25` retrieval methods with augmented generation based on the top-20 segments, you can run the following command:
```bash
python src/ragnarok/scripts/run_ragnarok.py  --model_path=command-r-plus  --topk=20 \
  --dataset=rag24.researchy-dev  --retrieval_method=bm25 --prompt_mode=cohere  \
  --context_size=8192 --max_output_tokens=1024 
```

Or to run the `gpt-4o` model on the `rag24.raggy-dev` dataset with `bm25` followed by `rank_zephyr_rho` retrieval methods, with augmented generation based on the top-5 segments, you can run the following command:
```bash
python src/ragnarok/scripts/run_ragnarok.py  --model_path=gpt-4o  --topk=100,5 \
    --dataset=rag24.raggy-dev  --retrieval_method=bm25,rank_zephyr_rho --prompt_mode=chatqa  \
    --context_size=8192 --max_output_tokens=1024  --use_azure_openai
```

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

TODO

## 🙏 Acknowledgments

This research is supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.
