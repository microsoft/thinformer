# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**kvpress** is a library for efficiently compressing the key-value (KV) cache of pretrained transformers during long-context inference. It implements multiple training-free KV cache compression methods and provides benchmarking tools using 🤗 transformers.

The repository contains:
- `kvpress/` - Core library implementing compression methods ("presses")
- `evaluation/` - Benchmarking framework and datasets
- `tests/` - Unit and integration tests
- `notebooks/` - Example notebooks and tutorials

## Development Commands

### Environment Setup
```bash
# Install with all dev dependencies using uv
uv sync --all-groups

# Install with optional dependencies
uv sync --extra flash-attn  # For flash attention support
uv sync --extra eval        # For evaluation benchmarks
```

### Code Quality
```bash
# Format code (runs isort + black)
make format

# Run style checks (flake8, mypy, SPDX headers)
make style

# Individual formatters
make isort    # Sort imports
make black    # Format code
```

### Testing
```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_pipeline.py -v

# Run with coverage
uv run pytest --cov=kvpress/ tests/

# Run single test
uv run pytest tests/test_pipeline.py::test_name -v
```

### Evaluation
```bash
# Navigate to evaluation directory first
cd evaluation

# Run evaluation with config file
python runtime.py --config_file evaluate_config.yaml

# Override config via command line
python runtime.py --dataset loogle --data_dir shortdep_qa \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --press_name expected_attention --compression_ratio 0.5

# Multi-GPU evaluation
bash evaluate.sh
```

## Architecture

### Press System (Core Abstraction)

All compression methods inherit from `BasePress` ([kvpress/presses/base_press.py](kvpress/presses/base_press.py)):

1. **BasePress** - Abstract base class
   - `compress()` - Core compression logic (must be implemented by subclasses)
   - `forward_hook()` - Hook registered to attention layers during prefilling
   - `__call__()` - Context manager that registers/unregisters hooks
   - `reset()` - Reset internal state between different contexts/questions

2. **ScorerPress** - Subclass for score-based pruning methods
   - Implements `score()` method that assigns importance to each KV pair
   - Common base for methods like SnapKV, KnormPress, ExpectedAttentionPress, etc.

3. **Wrapper Presses** - Compose or modify other presses
   - `ComposedPress` - Chain multiple presses together
   - `AdaKVPress` - Apply head-wise compression to any ScorerPress
   - `DecodingPress` - Compress during token generation (not just prefilling)
   - `PrefillDecodingPress` - Compress in both prefilling and decoding phases

### Pipeline Architecture

**KVPressTextGenerationPipeline** ([kvpress/pipeline.py](kvpress/pipeline.py)) is the main interface:

1. **Three-phase execution**:
   - `preprocess()` - Apply chat template, tokenize context and questions
   - `_forward()` - Compress context, then generate answers for each question
   - `postprocess()` - Return formatted results

2. **Key design decisions**:
   - Context compression is shared across all questions (one prefill pass)
   - Each question gets its own generation phase
   - Compression is NOT applied to question tokens (for fair benchmarking)
   - After each question, the answer is removed from cache via `_remove_answer_from_cache()`

3. **Timing infrastructure**:
   - `query_events` - Track CUDA timing for decoding (list of dicts, one per question)
   - `prefill_context_events` - Track context prefilling time (single dict, shared)
   - `prefill_question_events` - Track question prefilling time (list of dicts, one per question)
   - `cache_sizes_tracking` - Track cache size evolution during decoding

### Evaluation Framework

**runtime.py** ([evaluation/runtime.py](evaluation/runtime.py)) orchestrates benchmarking:

1. **Configuration hierarchy**:
   - Default values from `EvaluationConfig` dataclass
   - Override with YAML config file (`evaluate_config.yaml`)
   - Override with command-line arguments (highest priority)

2. **Execution flow**:
   ```
   _setup_model_pipeline() → _setup_press() → _load_and_prepare_dataset()
   → _run_inference() → _save_results() → _calculate_and_save_metrics()
   ```

3. **Warmup + Timed runs**:
   - `NUM_WARMUP` iterations to warm up GPU before timing
   - Separate tracking for warmup vs. timed runs
   - Records: query times, compression times, prefill times (context + question)

4. **Output structure**:
   ```
   results/
     └── {dataset}__{model}__{press}__{target_size}/
         ├── predictions.parquet  # Per-sample predictions and timings
         ├── metrics.json         # Aggregate metrics
         ├── config.yaml          # Full configuration
         └── times/
             └── times.json       # Detailed timing breakdown
   ```

### Registry System

**evaluate_registry.py** defines three registries:
- `PRESS_REGISTRY` - Maps press names to press instances
- `DATASET_REGISTRY` - Maps dataset names to HuggingFace dataset paths
- `SCORER_REGISTRY` - Maps dataset names to scoring functions

## Important Patterns

### Adding a New Press

1. Inherit from `BasePress` or `ScorerPress`
2. Implement `compress()` method (or `score()` for ScorerPress)
3. Add SPDX header: `# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
4. Register in `evaluate_registry.py` for benchmarking
5. Add tests in `tests/presses/`
6. See [notebooks/new_press.ipynb](notebooks/new_press.ipynb) for step-by-step guide

### Context Manager Pattern

Presses use context managers to temporarily register hooks:
```python
with press(model):
    # Compression is active during this block
    outputs = model(inputs)
# Hooks are automatically unregistered
```

### CUDA Event Timing

The codebase uses CUDA events for GPU timing:
```python
events = defaultdict(list)  # layer_idx -> [(start_event, end_event), ...]
outputs = model(..., query_events=events)
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

### Multi-Question Handling

- Context is prefilled once with compression
- For each question:
  1. Forward pass with question tokens (tracked separately)
  2. Autoregressive decoding
  3. Remove answer from cache using `_remove_answer_from_cache()`
  4. Cache is restored to post-context state for next question

## Code Style Requirements

1. **SPDX headers**: All Python files must include SPDX copyright header
2. **Line length**: 120 characters (enforced by black)
3. **Import order**: Use isort with 3-line mode
4. **Type hints**: Encouraged but not strictly enforced (mypy runs with `ignore_missing_imports`)
5. **Commit sign-off**: All commits must be signed off (`git commit -s`)

## Testing Philosophy

- Unit tests for individual presses in `tests/presses/`
- Integration tests for pipeline behavior in `tests/integration/`
- Decoding compression tests in `tests/test_decoding_compression.py`
- All tests must pass - skipped tests fail CI
- Coverage reports generated in `reports/coverage.xml`

## Special Considerations

### Model Support
Tested with: LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Gemma3ForConditionalGeneration. Other models may work if they follow similar architectures in transformers.

### Flash Attention
Some presses (e.g., ObservedAttentionPress) require eager attention implementation. Flash Attention 2 is auto-detected and used when available, except when explicitly disabled.

### Quantization
Supports transformers `QuantizedCache` (requires additional deps like `optimum-quanto`). Pass cache object to pipeline.

### Timing vs. Compression Trade-offs
- Prefill compression reduces decoding time (smaller cache → faster attention)
- Some presses (e.g., KVzipPress) require multiple forward passes for compression
- Decoding compression adds overhead during generation but controls memory growth

## Distributed Evaluation

Use SLURM for multi-node evaluation. Scripts in `evaluation/`:
- Activate venv in script: `source ../.venv/bin/activate`
- SLURM jobs start with clean environment (no inherited venv)
- Example: `launch-longbench-e.sh`, `launch-math500.sh`
