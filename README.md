# Attention-based NTP end to end training stack

End-to-end LLM training stack with Ray-based data preprocessing, tokenizer training, PyTorch training/evaluation, checkpointing, and experiment tooling.

## What this repo does

This project is an end-to-end LLM training workflow that covers:

- large-scale text data preprocessing
- tokenizer creation and dataset artifact generation
- PyTorch training and evaluation
- checkpointing and run metadata
- experiment tracking
- Triton/performance experimentation

The repo is structured around the full training stack rather than only model code.

## Current architecture

```text
Raw text / source dataset
        ↓
Ray-based preprocessing pipeline
(canonicalize → minhash → pairs → cluster_map → snapshot → split)
        ↓
Training-ready dataset artifacts
(train.bin / val.bin / metadata / vocab)
        ↓
Tokenizer loading + config-driven training
        ↓
PyTorch training loop
(loss, eval, checkpoints, benchmark hooks)
        ↓
Evaluation, logs, artifacts, experiment tracking
```

## Repository structure

- `dataset_pipeline/` — staged preprocessing pipeline
- `configs/` — configuration files for preprocess / tokenizer / train / eval / serve
- `main/` — training entrypoint, checkpointing, and run logic
- `eval/` — evaluation utilities
- `serve/` — serving/inference-related code
- `test_triton/` — Triton kernel tests and experiments
- `artifacts/` — generated outputs and pipeline artifacts
- `logs/` — runtime logs
- `research/` — research and experiment work
- `.dvc/` and related files — dataset/versioning workflow
- `.github/workflows/` — CI / automation

## Training stack highlights

### Data preprocessing

The preprocessing system is built as a staged pipeline using Ray and currently includes:

- canonicalize
- minhash
- pairs
- cluster_map
- snapshot
- split

This makes the data flow explicit and gives clean stage boundaries for debugging, reruns, and future observability work.

### Training

The training stack includes:

- config-driven PyTorch training
- checkpoint save/load helpers
- W&B integration
- MLflow hooks
- structured logging
- benchmark evaluation hooks

### What I am improving now

The next upgrades for this repo are focused on training-systems engineering:

- runtime observability across preprocessing and training
- checkpoint/resume validation
- throughput benchmarking and bottleneck diagnosis
- reliability and failure handling across long-running runs
- stronger reproducibility documentation

## Why this project exists

The goal of this repo is not just to train a model, but to understand the full system around training: data ingestion, preprocessing, tokenizer artifacts, training runs, checkpoints, evaluation, and performance behavior.

## Next milestones

- add system metrics such as step time, tokens/sec, and stage throughput
- validate checkpoint resume correctness
- write bottleneck reports for long-running runs
- improve documentation for reproducibility and debugging



## Runtime observability

A major focus of this project is treating model training as a system, not just a loss curve.

The next step for this repo is improving visibility into how preprocessing and training behave over long runs so bottlenecks, regressions, and failures can be diagnosed more systematically.

### Metrics I want to track

#### Preprocessing pipeline

For the Ray-based preprocessing stages, the main observability targets are:

- rows or records processed per second
- tokens produced per second
- per-stage elapsed time
- load time
- transform or tokenization time
- write time
- skipped or malformed record count
- retry and error counts
- shard or worker-level progress where possible

This should make it easier to identify whether bottlenecks are coming from ingestion, transformation, serialization, or artifact writing.

#### Training pipeline

For the PyTorch training loop, the main observability targets are:

- step time
- tokens per second
- samples per second
- data loading time
- forward and backward pass time
- optimizer step time
- evaluation overhead
- checkpoint save and load time
- gradient norm
- NaN or Inf detection
- GPU memory allocated and reserved

This should make it easier to distinguish between model-quality metrics and system-health metrics.

### Why this matters

The goal is not just to see whether loss goes down, but to understand how the full training stack behaves:

- whether the input pipeline is starving training
- whether throughput degrades over time
- whether checkpointing interrupts runtime too heavily
- whether evaluation causes stalls
- whether memory usage drifts across long runs
- whether failures can be diagnosed from logs and metrics rather than guesswork

### Observability goals

The observability work in this repo is intended to support:

- bottleneck diagnosis across data, training, and checkpointing stages
- more reliable long-running runs
- easier debugging of failures and regressions
- clearer comparisons across scaling-law experiments
- stronger reproducibility and training-systems discipline

### Next implementation steps

- log preprocessing stage timings and throughput
- log training runtime metrics to W&B
- add run summaries for throughput and timing
- validate checkpoint and resume behavior under interruption
- write benchmark reports identifying likely bottlenecks


How long is each step taking?

How many tokens/sec am I getting?

Is the GPU actually busy?

Is it waiting for data?

Is memory stable over time?