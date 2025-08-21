# Conflict-Aware Soft Prompting for Retrieval-Augmented Generation

This repository contains the official implementation for our EMNLP 2025 paper:

> **Conflict-Aware Soft Prompting for Retrieval-Augmented Generation**  
> Eunseong Choi, June Park, Hyeri Lee, and Jongwuk Lee  
> *Accepted to EMNLP 2025*

---

## Overview

Retrieval-augmented generation (RAG) enhances the capabilities of large language models (LLMs) by incorporating external knowledge into their input prompts. However, when the retrieved context contradicts the parametric knowledge of LLMs, it often fails to resolve the conflict between incorrect external context and correct parametric knowledge, known as context-memory conflict. To tackle this problem, we introduce **Conflict-Aware REtrieval-Augmented Generation (CARE)**, consisting of a context assessor and a base LLM. The context assessor encodes compact memory token embeddings from raw context tokens. Through grounded/adversarial soft prompting, the context assessor is trained to discern unreliable context and capture a guidance signal that directs reasoning toward the more reliable knowledge source.

TBU
