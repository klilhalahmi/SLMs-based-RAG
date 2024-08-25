# Multiple SLMs based Retrieval-Augmented Generation (RAG)

This repository contains the implementation of a novel approach to Retrieval-Augmented Generation (RAG) that leverages multiple Small Language Models (SLMs) instead of a single Large Language Model (LLM). Our framework aims to evaluate this advanced RAG technique, focusing on balancing performance with computational efficiency.

## Abstract

This framework introduces a novel approach to Retrieval-Augmented Generation (RAG) that leverages multiple Small Language Models (SLMs). Our method employs three distinct SLMs for generating embeddings and creating individual vector databases, followed by the application of Reciprocal Rank Fusion (RRF) and CombSUM techniques to aggregate and rank the results. We evaluated our approach using the Scifact corpus, comparing the performance of four embedding models (three SLMs and one LLM) and two ensemble methods across multiple metrics.

## Introduction

Retrieval-Augmented Generation (RAG) enhances text generation by integrating retrieval mechanisms with generative models. This study addresses the inherent performance and computational trade-offs associated with using large language models (LLMs) in RAG pipelines by proposing a multi-SLM approach.

## Background

- **RAG (Retrieval-Augmented Generation):** Enhances text generation by incorporating retrieval mechanisms.
- **SLMs (Small Language Models):** More computationally efficient than LLMs, although they typically perform worse on individual tasks.
- **Ensemble Methods:** Combining multiple models can yield better performance than a single large model.

## Our Approach

Our project introduces a multi-SLM pipeline utilizing:

- Three distinct SLMs from Hugging Face for embedding generation.
- Reciprocal Rank Fusion (RRF) and CombSUM techniques for result aggregation and ranking.
- A comprehensive framework for evaluating these techniques.


## Framework

The framework is highly configurable via a `configs.yml` file and includes the following modules:

1. `tinies_vs_gaint.py`: Acts as the main manager for the entire process.
2. `eval.py`: Contains all evaluation metrics.
3. `fusion.py`: Incorporates all fusion methods.
4. `index_manager`: Handles indexing and searching operations.
5. `logger`: Custom debugging tool.

## Configuration

The `configs.yml` file enables configuration of:

- **slm_models_names**: Names of the SLMs to be used.
- **llm_model_name**: Name of the LLM for comparison.
- **datasets_names**: Lists of datasets on which the models will be evaluated.
- **output_folder**: Defines the output folder for storing all results.
- **save_index**: Option to save and load vector database indexes.

## Experimental Results

### Evaluation Metrics

1. **Average Precision (AP)**
2. **Precision at 3 (P@3)**
3. **Precision at 5 (P@5)**
4. **Recall**

### Model Performance example

| Model                                      | AP    | P@3   | P@5   | Recall |
|--------------------------------------------|-------|-------|-------|--------|
| snowflake-arctic-embed-m-v1.5              | 0.314 | 0.129 | 0.093 | 0.513  |
| privacy_embedding_rag_10k_base_12_final    | 0.379 | 0.159 | 0.109 | 0.578  |
| bert-base-uncased                          | 0.111 | 0.043 | 0.037 | 0.226  |
| gte-Qwen2-1.5B-instruct                    | 0.434 | 0.177 | 0.126 | 0.647  |
| **RRF**                                    | 0.432 | 0.179 | 0.123 | 0.705  |
| **CombSum**                                | 0.434 | 0.176 | 0.119 | 0.705  |

### Indexing Times

| Model                                    | Indexing Time (seconds) |
|------------------------------------------|-------------------------|
| snowflake-arctic-embed-m-v1.5            | 30.01                   |
| privacy_embedding_rag_10k_base_12_final  | 28.56                   |
| bert-base-uncased                        | 28.95                   |
| gte-Qwen2-1.5B-instruct                  | 369.90                  |

## Discussion

- **Trade-offs**: Larger models offer better performance but at a higher computational cost.
- **Ensemble Methods**: Both RRF and CombSum improve overall performance, especially in recall.
- **SLM Efficiency**: SLMs are faster and can be more practical for real-time applications, especially when combined through ensemble methods.

## Future Work

- Explore more sophisticated ensemble techniques.
- Investigate fine-tuning SLMs for specific tasks.
- Optimize indexing for larger models.
- Experiment with hyperparamters for fusion
- Experiment with different embedding techniques for large texts


## References

For detailed technical references, please refer to the paper and the accompanying bibliography in the documentation.

---

Feel free to navigate the repository and experiment with different configurations to better understand the trade-offs and performance gains associated with multi-SLM RAG systems.
