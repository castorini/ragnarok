# RAGnarok

<!-- [![PyPI](https://img.shields.io/pypi/v/rank-llm?color=brightgreen)](https://pypi.org/project/rank-llm/) -->
<!-- [![Downloads](https://static.pepy.tech/personalized-badge/rank-llm?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/rank-llm) -->
<!-- [![Downloads](https://static.pepy.tech/personalized-badge/rank-llm?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/week)](https://pepy.tech/project/rank-llm) -->
<!-- [![Generic badge](https://img.shields.io/badge/arXiv-2309.15088-red.svg)](https://arxiv.org/abs/2309.15088) -->
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)


A battleground (RAGnarok) for the best retrieval-augmented generation models!

# Releases
current_version = 0.0.0

## 📟 Instructions

### Create Conda Environment

```bash
conda create -n ragnarok python=3.10
conda activate ragnarok
```

### Install Pytorch with CUDA
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # For CUDA 11.8
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run generation test
```bash
python src/ragnarok/scripts/run_ragnarok.py  --model_path=cohere-commandr-plus \
  --top_k_candidates=20 
  --dataset=dl-raggy-topics \
  --context_size=4096
```

### Contributing 

If you would like to contribute to the project, please refer to the [contribution guidelines](CONTRIBUTING.md).

## 🦙🐧 Model Zoo

Most LLMs supported by FastChat should additionally be supported by RAGnarok too, albeit we do not test all of them. If you would like to see a specific model added, please open an issue or a pull request. The following is a table of models *tested* by RAGnarok.:

| Model Name        | Identifier/Link                            |
|-------------------|---------------------------------------------|
| LLaMA



## ✨ References

If you use RAGnarok, please cite the following:

## 🙏 Acknowledgments

This research is supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.
