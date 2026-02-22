# AutoFT-Builder

## Intelligent Fine-Tuning Dataset Generator with Semantic Deduplication

------------------------------------------------------------------------

# 1. Overview

## 1.1 Product Summary

AutoFT-Builder is a lightweight backend pipeline that generates
high-quality synthetic fine-tuning datasets using large language models
(LLMs) while automatically filtering out semantically redundant samples
using embedding similarity.

The system ensures diversity and reduces harmful duplication in
fine-tuning datasets by leveraging:

-   LLM-based data generation (via OpenRouter API)
-   Local embedding model (BAAI/bge-small-en-v1.5)
-   Cosine similarity filtering
-   Persistent dataset storage
-   JSONL export for fine-tuning

------------------------------------------------------------------------

## 1.2 Problem Statement

Synthetic data generation using LLMs often results in:

-   Near-duplicate samples
-   Structural repetition
-   Low semantic diversity
-   Overfitting risk during fine-tuning

There is no lightweight pipeline that:

1.  Generates synthetic data
2.  Automatically filters semantically similar samples
3.  Stores only diverse entries
4.  Exports fine-tuning-ready datasets

------------------------------------------------------------------------

## 1.3 Goal

Build a CLI-based pipeline that:

-   Generates synthetic training data in batches
-   Embeds each sample locally
-   Filters semantically similar entries
-   Stores only diverse examples
-   Exports fine-tuning-ready JSONL files

------------------------------------------------------------------------

# 2. Scope

## 2.1 In Scope

-   LLM-based synthetic dataset generation
-   Local embedding using BAAI/bge-small-en-v1.5
-   Cosine similarity filtering
-   Batch-based processing
-   SQLite dataset storage
-   JSONL export
-   Configurable similarity threshold

## 2.2 Out of Scope (MVP)

-   Web UI
-   Distributed system support
-   Multi-user support
-   Real-time monitoring dashboard
-   Human annotation workflows
-   Reinforcement learning

------------------------------------------------------------------------

# 3. High-Level Architecture

User → Generate Batch → Temporary Storage → Embedding → Similarity Check
→ Filter → Store in Dataset DB → Export JSONL

------------------------------------------------------------------------

# 4. Functional Requirements

## 4.1 Data Generation Module

### Input

-   Task description (string)
-   Output format specification
-   Batch size (int)
-   Optional: style diversification instructions

### Output Format (Standardized)

{ "instruction": "...", "output": "..." }

------------------------------------------------------------------------

## 4.2 Embedding Module

-   Model: BAAI/bge-small-en-v1.5
-   Prefix required: "Represent this sentence for searching relevant
    passages:"

Embedding target: instruction + " " + output

Output: - 384-dimensional float vector

------------------------------------------------------------------------

## 4.3 Similarity Filtering Module

Algorithm: Cosine similarity

For each new sample: 1. Compare against existing dataset 2. Compare
against accepted samples in current batch

If similarity \> threshold → Discard Else → Accept

Default threshold: 0.85 Recommended range: 0.80 -- 0.90

------------------------------------------------------------------------

## 4.4 Storage Layer

Database: dataset.db (SQLite)

Table Schema:

## samples

id (INTEGER PRIMARY KEY) instruction (TEXT) output (TEXT) embedding
(BLOB) created_at (TIMESTAMP)

------------------------------------------------------------------------

## 4.5 Export Module

Command:

python main.py export --output dataset.jsonl

Output format:

{"instruction": "...", "output": "..."}

------------------------------------------------------------------------

# 5. Non-Functional Requirements

-   Must run on CPU-only systems
-   Must handle at least 10,000 samples
-   Must operate under 1GB RAM
-   Must support incremental dataset building
-   Must not store duplicate entries

------------------------------------------------------------------------

# 6. Technical Stack

-   Python 3.10+
-   OpenRouter API
-   sentence-transformers
-   BAAI/bge-small-en-v1.5
-   SQLite
-   numpy
-   argparse
-   tqdm

------------------------------------------------------------------------

# 7. Definition of Done

The system is complete when:

-   It can generate and store at least 1,000 diverse samples
-   No duplicate or near-duplicate entries are present
-   JSONL export works correctly
-   Dataset is directly usable in fine-tuning frameworks
-   Entire pipeline runs locally on CPU

------------------------------------------------------------------------

# One-Sentence Product Pitch

A lightweight pipeline that leverages frontier LLMs to generate
high-quality synthetic fine-tuning data while automatically enforcing
semantic diversity using embedding similarity filtering.
