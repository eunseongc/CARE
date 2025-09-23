# Conflict-Aware Soft Prompting for Retrieval-Augmented Generation (CARE)

This repository contains the official implementation for our EMNLP 2025 paper:

> **Conflict-Aware Soft Prompting for Retrieval-Augmented Generation**

> Eunseong Choi, June Park, Hyeri Lee, and Jongwuk Lee

> *EMNLP 2025*

---

## Overview

Retrieval-augmented generation (RAG) enhances the capabilities of large language models (LLMs) by incorporating external knowledge into their input prompts. However, when the retrieved context contradicts the parametric knowledge of LLMs, it often fails to resolve the conflict between incorrect external context and correct parametric knowledge, known as **context-memory conflict**.

To address this, we propose **Conflict-Aware REtrieval-Augmented Generation (CARE)**, which consists of:

* **Context Assessor**: Encodes compact *memory token* embeddings from raw context tokens and learns to assess the reliability of external evidence.
* **Base LLM**: Receives grounded/adversarial soft prompts from the assessor to better resolve conflicts between external context and parametric memory.

Through this design, CARE mitigates the harmful influence of unreliable retrieved context while leveraging reliable evidence effectively.

---

## 1. Environment Setup

We recommend using Docker for reproducibility. Example setup:

```bash
export env_name=care_env
export home_dir=/path/to/workspace

docker run --gpus '"device=0,1"' --shm-size=8G -it \
    -v ${home_dir}:/workspace \
    --name ${env_name} \
    pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Attach to container
docker exec -it ${env_name} bash
```

Inside the container:

```bash
git clone https://github.com/eunseongc/CARE
cd CARE
pip install -r requirements.txt
```

---

## 2. Data and Checkpoints

We provide preprocessed data and model checkpoints via Hugging Face.

* **Finetuned checkpoints**

  * [care\_mistral](https://huggingface.co/eunseong/care_mistral)
  * [care\_llama](https://huggingface.co/eunseong/care_llama)
  * [care\_qwen](https://huggingface.co/eunseong/care_qwen)

* **Pretraining checkpoints**

  * [care\_mistral\_pt](https://huggingface.co/eunseong/care_mistral_pt)
  * [care\_llama\_pt](https://huggingface.co/eunseong/care_llama_pt)
  * [care\_qwen\_pt](https://huggingface.co/eunseong/care_qwen_pt)

* **Datasets**

  * [data\_care](https://huggingface.co/datasets/eunseong/data_care) → required (place inside `CARE/data_care/`)
  * [nq\_colbertv2](https://huggingface.co/datasets/eunseong/nq_colbertv2) → optional (used in `data_preprocess.ipynb`)

**Directory structure for data:**

```
CARE
└── data_care/
    ├── eval/
    ├── finetune/
    └── pretrain/
```

**Note on corpus**: Following [xRAG](https://github.com/Hannibal046/xRAG), we use the **Wikipedia dump from December 2021**(https://github.com/facebookresearch/atlas?tab=readme-ov-file#models) as the knowledge source.
* Pretraining corpus: 2 million samples randomly selected.

---

## 3. Training & Evaluation

### Finetuning

```bash
source finetune.sh mistral   # or llama / qwen
```

### Evaluation

```bash
source evaluate.sh
```

* `evaluate.sh`: Standard evaluation
* `evaluate_closedbook.sh`: Closed-book evaluation

---

## 4. Pretraining (Optional)

Pretraining CARE is computationally expensive and only needed if you want to train from scratch.

* On **2× A100 GPUs**, pretraining takes about **25 hours**.

```bash
source pretrain.sh mistral   # or llama / qwen
```

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{choi2025care,
  author    = {Eunseong Choi and
               June Park and
               Hyeri Lee and
               Jongwuk Lee},
  title     = {Conflict-Aware Soft Prompting for Retrieval-Augmented Generation},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  publisher = {Association for Computational Linguistics},
  year      = {2025}
}
```
