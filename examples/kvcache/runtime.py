# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
# from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from datasets import load_dataset
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import Pipeline, pipeline

from kvpress import ComposedPress, DuoAttentionPress, FinchPress, ObservedAttentionPress, ScorerPress, ThinKPress
from kvpress.presses.decoding_press import DecodingPress

from thinformer.timing_utils import multi_events

logger = logging.getLogger(__name__)

NUM_WARMUP = int(os.environ.get("NUM_WARMUP", 1))     # Number of warmup iterations to skip during inference


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "longbench-e"
    data_dir: Optional[str] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[str] = None
    press_name: str = "knorm"
    compression_ratio: float = 1.0
    key_channel_compression_ratio: Optional[float] = None
    do_sample: bool = False

    # Dataset and generation parameters
    fraction: float = 1.0
    batch_size: int = 0  # number of samples to process in a job, 0 means no batching
    batch_number: int = 0  # batch number to process, 0 means no batching
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    compress_questions: bool = False
    needle_depth: Optional[int] = None

    # Decoding parameters
    compression_interval: Optional[int] = None
    target_size: Optional[int] = None
    hidden_states_buffer_size: Optional[int] = None

    # Output and logging
    output_dir: str = "./results"
    log_level: str = "INFO"

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42
    force: bool = False

    # For Express
    h: Optional[int] = None

    # Patch the model's attention with the custom (Express-prefill) implementation.
    # Enabled for the accelerating-KV-cache experiment; disabled for long-form decoding.
    use_express_prefill: bool = False
    # Skip the tokenizer chat template (some datasets score better without it).
    disable_chat_template: bool = False
    # Enable PyramidKV's causal_lower_right attention mask.
    use_pyramidkv: bool = False
    # Express attention sink / sliding-window sizes.
    n_first: int = 0
    n_last: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"

        if self.press_name == "no_press":
            # override compression_ratio to 0.0
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0

        # Validate compression ratios
        assert (
            0.0 <= self.compression_ratio <= 1.0
        ), f"compression_ratio must be between 0.0 and 1.0, got {self.compression_ratio}"

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"

        # Initialize model_kwargs if None
        if self.model_kwargs is None:
            self.model_kwargs = {}

        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert self.max_context_length is not None, "max_context_length must be set for needle_in_haystack"

    def get_results_dir(self, output_dir: Path) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path
        press
            The press instance to check for ThinKPress components

        Returns
        -------
        Path
            The path to the results directory
        """
        # Build directory name components
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir is not None else "",
            self.model.replace("/", "--"),
            self.press_name,
            f"{self.target_size}",
            f"h{self.h}",
        ]

        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.batch_size > 0 and self.batch_number > 0:
            components.append(f"bs{self.batch_size}_bn{self.batch_number}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.compress_questions:
            components.append("compressed_questions")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")
        if self.seed is not None:
            components.append(f"seed{self.seed}")

        dir_name = "__".join(filter(None, components))  # Filter None/empty strings
        config_dir = output_dir / dir_name

        # Make sure the directory does not exist, if it does, add a number to the end
        # This is to avoid overwriting results
        # if config_dir.exists():
        #     i = 1
        #     while (config_dir / f"{i}").exists():
        #         i += 1
        #     config_dir = config_dir / f"{i}"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def save_config(self, config_filename: Path):
        """
        Saves the evaluation configuration to a YAML file.
        """
        with open(str(config_filename), "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path) -> dict:
    """Loads a YAML file. Returns an empty dict if it doesn't exist."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using only command-line arguments and defaults.")
        return {}


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : EvaluationConfig
        The configuration for the evaluation run.

    The final output will be predictions_<config>.csv and metrics_<config>.json in the output_dir.
    If the evaluation files already exist, evaluation will be skipped.

    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the EvaluationRunner with a given configuration.

        Parameters
        ----------
        config : EvaluationConfig
            The configuration for the evaluation run.
        """
        self.config = config
        self.pipeline: Optional[Pipeline] = None  # Will be set by _setup_model_pipeline()
        self.press: None | ScorerPress = None  # Will be set by _setup_press()
        self.df: Optional[pd.DataFrame] = None  # Will be set by _load_dataset()
        self._setup_logging()
        # self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        """Set deterministic seeds for reproducible results."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug(f"Set deterministic seeds to {self.config.seed}")

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.log_level.upper()

        # Configure the root logger so all modules get the same log level
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        
        # Set level on root logger to affect all loggers
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)
        
        # Also set level on this module's logger
        logger.setLevel(log_level)

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.

        Returns
        -------
        Path
            The path to the output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        target_size = self.config.target_size
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]

        # Apply compression ratios based on press type
        if isinstance(press, DuoAttentionPress):
            press.head_compression_ratio = compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {compression_ratio}")
        elif isinstance(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    assert (
                        key_channel_compression_ratio is not None
                    ), "key_channel_compression_ratio must be set for ThinKPress in ComposedPress"
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                    logger.info(f"Set ComposedPress key_channel_compression_ratio to {key_channel_compression_ratio}")
                else:
                    # Check if compression_ratio attribute exists before setting
                    if hasattr(ps, "compression_ratio"):
                        ps.compression_ratio = compression_ratio
                        logger.info(f"Set ComposedPress compression_ratio to {compression_ratio}")
                    else:
                        logger.warning(
                            f"ComposedPress component {ps.__class__.__name__} has no 'compression_ratio' attribute."
                        )
        elif isinstance(press, ThinKPress):
            assert key_channel_compression_ratio is not None, "key_channel_compression_ratio must be set for ThinKPress"
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")
        elif isinstance(press, DecodingPress):
            press.compression_interval = self.config.compression_interval or press.compression_interval
            press.target_size = self.config.target_size or press.target_size
            press.hidden_states_buffer_size = self.config.hidden_states_buffer_size or press.hidden_states_buffer_size
            logger.info(
                f"Set DecodingPress compression_interval to {self.config.compression_interval}, target_size to {self.config.target_size}, hidden_states_buffer_size to {self.config.hidden_states_buffer_size}"
            )
        else:
            if hasattr(press, "target_size"):
                press.target_size = target_size
                logger.info(f"Set {press.__class__.__name__} target_size to {target_size}")
            else:
                logger.warning(
                    f"Press {press.__class__.__name__} has no 'target_size' attribute. This is expected is you set `no_press`."
                )

        # Accelerating KV Cache Compression prunes to a fixed target_size instead
        # of a compression ratio when enabled (see ScorerPress.compress).
        if hasattr(press, "use_express_prefill"):
            press.use_express_prefill = self.config.use_express_prefill

        self.press = press
        # Set the press info in the config for saving to YAML
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")

    def _load_and_prepare_dataset(self):
        """
        Loads the dataset specified in the config and applies sampling/filtering.
        """
        dataset_name = self.config.dataset
        data_dir = str(self.config.data_dir) if self.config.data_dir else None
        fraction = self.config.fraction

        logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")
        df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()

        if fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=1)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        if self.config.batch_size > 0 and self.config.batch_number > 0:
            # assert fraction == 1.0, "Batching is only supported for full dataset"
            original_len = len(df)
            start = (self.config.batch_number - 1) * self.config.batch_size
            end = start + self.config.batch_size
            df = df.iloc[start:end]
            logger.info(f"Select samples {start} to {end} (exclusive) from {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        # # if we have needle in a haystack, we need to insert it in the context
        # if self.config.dataset == "needle_in_haystack":
        #     df = insert_needle_in_haystack(
        #         df, self.pipeline.tokenizer, self.config.max_context_length, self.config.needle_depth
        #     )

        if isinstance(self.press, FinchPress):
            if not self.config.compress_questions:
                logger.error("FinchPress requires 'compress_questions' to be set to True.")
                raise ValueError("FinchPress requires compress_questions to be set to True")
            # FinchPress uses a delimiter token to separate context and question
            # So we need to update the tokenizer and the model embeddings.
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(
                self.pipeline.model, self.pipeline.tokenizer
            )  # type: ignore[attr-defined]
            df["context"] = df["context"] + self.press.delimiter_token  # type: ignore[attr-defined, index]

        if self.config.compress_questions:
            logger.info("Compressing questions into context.")
            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")

        model_kwargs = self.config.model_kwargs or {}
        if isinstance(self.press, ObservedAttentionPress):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("ObservedAttentionPress detected, setting attn_implementation to 'eager'.")
        else:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")
                pass

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")
        pipeline_kwargs = {
            "model": model_name,
            "model_kwargs": model_kwargs,
            "trust_remote_code": True,
        }
        if device == "auto":
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device
        self.pipeline = pipeline("kv-press-text-generation", **pipeline_kwargs)

        if self.config.use_express_prefill:
            # Patch Llama attention layers with custom implementation for timing
            from modeling_llama import LlamaAttention
            for layer_idx, layer in enumerate(self.pipeline.model.model.layers):
                original_attn = layer.self_attn
                custom_attn = LlamaAttention(
                    self.pipeline.model.config,
                    layer_idx,
                    h=self.config.h,
                    use_pyramidkv=self.config.use_pyramidkv,
                    n_first=self.config.n_first,
                    n_last=self.config.n_last,
                )
                custom_attn.load_state_dict(original_attn.state_dict())
                custom_attn = custom_attn.to(
                    device=original_attn.q_proj.weight.device,
                    dtype=original_attn.q_proj.weight.dtype
                )
                layer.self_attn = custom_attn
            logger.info("Patched Llama attention layers with custom implementation.")

        self.pipeline.model.eval()
        logger.info("Model pipeline loaded.")

    @torch.inference_mode()
    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """

        self.df["predicted_answer"] = None  # type: ignore[index]
        self.df["cache_sizes"] = None  # type: ignore[index]
        self.df["cache_sizes_tracking"] = None  # type: ignore[index]
        self.df["compression_times"] = None  # type: ignore[index]
        self.df["query_times"] = None  # type: ignore[index]
        self.df["prefill_context_times"] = None  # type: ignore[index]
        self.df["prefill_question_times"] = None  # type: ignore[index]
        self.df["warm_query_times"] = None  # type: ignore[index]
        self.df["warm_compression_times"] = None  # type: ignore[index]
        self.df["warm_prefill_context_times"] = None  # type: ignore[index]
        self.df["warm_prefill_question_times"] = None  # type: ignore[index]

        logger.info("Running inference for each context-question pair.")
        times = {}
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
            context = row["context"]
            question = row["question"]
            answer_prefix = row["answer_prefix"]
            max_new_tokens = self.config.max_new_tokens or row["max_new_tokens"]
            
            warm_query_times = []
            warm_compression_times = []
            warm_prefill_context_times = []
            warm_prefill_question_times = []

            for i in range(NUM_WARMUP):
                multi_events.reset()
                self._setup_deterministic_seeds()
                warm_query_events = defaultdict(list)
                warm_prefill_context_events = defaultdict(list)
                warm_prefill_question_events = defaultdict(list)
                warm_output = self.pipeline(
                    context,
                    question=question,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                    do_sample=self.config.do_sample,
                    disable_chat_template=self.config.disable_chat_template,
                    query_events=[warm_query_events,],  # self.pipeline expects a list of dictionaries
                    prefill_context_events=warm_prefill_context_events,
                    prefill_question_events=[warm_prefill_question_events,],  # list of dicts, one per question
                )
                torch.cuda.synchronize()
                # Compute query_times from query_events (mutable dict)
                warm_query_times_by_layer = {
                    layer_idx: [
                        start.elapsed_time(end) for start, end in events
                    ] for layer_idx, events in warm_query_events.items()
                }
                _, _, _ = warm_output["answer"]
                # add warm up query times to dictionary
                # Save the total query time across all layers for the warmup iteration
                warm_query_times.append(np.sum([np.sum(times) for k, times in warm_query_times_by_layer.items()]))

                # Compute prefill_context_times from prefill_context_events (mutable dict)
                warm_prefill_context_times_by_layer = {
                    layer_idx: [
                        start.elapsed_time(end) for start, end in events
                    ] for layer_idx, events in warm_prefill_context_events.items()
                }
                # Save the total prefill context time across all layers for the warmup iteration
                warm_prefill_context_times.append(np.sum([np.sum(times) for k, times in warm_prefill_context_times_by_layer.items()]))

                # Compute prefill_question_times from prefill_question_events (mutable dict)
                warm_prefill_question_times_by_layer = {
                    layer_idx: [
                        start.elapsed_time(end) for start, end in events
                    ] for layer_idx, events in warm_prefill_question_events.items()
                }
                # Save the total prefill question time across all layers for the warmup iteration
                warm_prefill_question_times.append(np.sum([np.sum(times) for k, times in warm_prefill_question_times_by_layer.items()]))

                # compute the total compression time (in milliseconds) across all layers
                # each layer has a list of (start, end) events
                if hasattr(self.press, 'events'):
                    compression_times = []
                    for events in self.press.events.values():
                        compression_times.append( np.sum([start.elapsed_time(end) for start, end in events]) )
                    warm_compression_times.append(np.sum(compression_times))
                else:
                    warm_compression_times.append(None)

                # Compute auxiliary timings
                time_dict = defaultdict(list)
                for event_name, events_list in multi_events.items():
                    time_dict[event_name] = [start.elapsed_time(end) for start, end in events_list]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory
                logger.debug(f"Warmup {i} times: {time_dict}")

            self.df.at[index, "warm_query_times"] = warm_query_times
            self.df.at[index, "warm_compression_times"] = warm_compression_times
            self.df.at[index, "warm_prefill_context_times"] = warm_prefill_context_times
            self.df.at[index, "warm_prefill_question_times"] = warm_prefill_question_times
            # Timed run
            multi_events.reset()
            self._setup_deterministic_seeds()
            query_events = defaultdict(list)
            prefill_context_events = defaultdict(list)
            prefill_question_events = defaultdict(list)
            cache_sizes_tracking = defaultdict(list)
            output = self.pipeline(
                context,
                question=question,
                answer_prefix=answer_prefix,
                press=self.press,
                max_new_tokens=max_new_tokens,
                max_context_length=self.config.max_context_length,
                do_sample=self.config.do_sample,
                disable_chat_template=self.config.disable_chat_template,
                query_events=[query_events,],  # self.pipeline expects a list of dictionaries
                prefill_context_events=prefill_context_events,
                prefill_question_events=[prefill_question_events,],  # list of dicts, one per question
                cache_sizes_tracking=cache_sizes_tracking,
            )
            torch.cuda.synchronize()
            # Compute query_times from query_events (mutable dict)
            query_times = {
                layer_idx: [
                    start.elapsed_time(end) for start, end in events
                ] for layer_idx, events in query_events.items()
            }
            # Compute prefill_context_times from prefill_context_events (mutable dict)
            prefill_context_times = {
                layer_idx: [
                    start.elapsed_time(end) for start, end in events
                ] for layer_idx, events in prefill_context_events.items()
            }
            # Compute prefill_question_times from prefill_question_events (mutable dict)
            prefill_question_times = {
                layer_idx: [
                    start.elapsed_time(end) for start, end in events
                ] for layer_idx, events in prefill_question_events.items()
            }
            answer, prompt_tokens, completion_tokens = output["answer"]
            self.df.loc[index, "predicted_answer"] = answer  # type: ignore[union-attr]
            self.df.loc[index, "prompt_tokens"] = prompt_tokens  # type: ignore[union-attr]
            self.df.loc[index, "completion_tokens"] = completion_tokens  # type: ignore[union-attr]
            # Cache size for KV caching
            if hasattr(self.press, "cache_size"):
                # compute mean real compression ratio across layers
                self.df.loc[index, "cache_size"] = np.mean(list(self.press.cache_size.values()))
            else:
                self.df.loc[index, "cache_size"] = None
            # Cache size for decoding
            if hasattr(self.press, 'cache_sizes') and self.press.cache_sizes:
                # TODO (Albert): use average cache size across layers if each layer has a different cache size
                self.df.loc[index, "mean_cache_size"] = np.mean(self.press.cache_sizes[0])
                self.df.at[index, "cache_sizes"] = self.press.cache_sizes[0]  # type: ignore[union-attr]
            else:
                self.df.loc[index, "mean_cache_size"] = None
                self.df.loc[index, "cache_sizes"] = None

            # Add compression times
            if hasattr(self.press, 'events') and self.press.events:
                # compute the total compression time (in milliseconds) across all layers
                # each layer has a list of (start, end) events
                compression_times = {
                    layer_idx: [
                        start.elapsed_time(end) for start, end in events
                    ] for layer_idx, events in self.press.events.items()
                }
                self.df.at[index, "compression_times"] = compression_times
            else:
                self.df.at[index, "compression_times"] = None

            # NOTE: the number of query times is one less than the value of completion_tokens
            # because we do not count the query time for the question
            self.df.at[index, "query_times"] = query_times  # type: ignore[union-attr]

            # Store prefill context times (context prefilling)
            self.df.at[index, "prefill_context_times"] = prefill_context_times  # type: ignore[union-attr]

            # Store prefill question times (question prefilling)
            self.df.at[index, "prefill_question_times"] = prefill_question_times  # type: ignore[union-attr]

            # Store cache sizes tracking
            self.df.at[index, "cache_sizes_tracking"] = cache_sizes_tracking  # type: ignore[union-attr]

            # Compute auxiliary timings
            time_dict = defaultdict(list)
            for event_name, events_list in multi_events.items():
                time_dict[event_name] = [start.elapsed_time(end) for start, end in events_list]
            logger.debug(f"Times: {time_dict}")
            times[index] = time_dict
            torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        # save times to a json file
        with open(self.times_dir / "times.json", "w") as f:
            json.dump(times, f, indent=4)

        logger.info("Inference completed.")

    def _save_results(self, save_filename: Path):
        """
        Saves the predicted answers and compression ratios to a CSV file.

        Parameters
        ----------
        save_filename : Path
            The full path including filename to save the CSV.
        """
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        self.df[list(set(self.df.columns) - set(["context"]))].to_parquet(
            str(save_filename), index=False
        )  # type: ignore[index]
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        """
        Calculates evaluation metrics and saves them to a JSON file.

        Parameters
        ----------
        save_filename : Path
            The base filename (e.g., CSV path) to derive the JSON path from.
        """
        dataset_name = self.config.dataset
        scorer = SCORER_REGISTRY[dataset_name]

        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        metrics = scorer(self.df)  # type: ignore[call-arg]
        # save cache size
        # metrics["target_size"] = self.df["target_size"].mean()  # type: ignore[index]
        metrics["cache_size"] = self.df["cache_size"].mean()  # type: ignore[index]
        metrics["cache_size_std"] = self.df["cache_size"].std()  # type: ignore[index]

        with open(str(save_filename), "w") as f:
            json.dump(metrics, f, indent=4)  # Pretty print JSON

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process.
        """
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()

        results_dir = self.config.get_results_dir(output_dir)
        self.times_dir = results_dir / "times"
        self.times_dir.mkdir(parents=True, exist_ok=True)
        predictions_filename = results_dir / "predictions.parquet"
        metrics_filename = results_dir / "metrics.json"
        config_filename = results_dir / "config.yaml"

        if predictions_filename.exists() and metrics_filename.exists():
            if self.config.force:
                logger.info(
                    f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nOverwriting..."
                )
            else:
                logger.info(
                    f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
                )
                return

        self._setup_press()
        self._setup_model_pipeline()
        self._load_and_prepare_dataset()

        self._run_inference()
        self._save_results(predictions_filename)
        self._calculate_and_save_metrics(metrics_filename)
        self.config.save_config(config_filename)
        logger.info("Evaluation run completed successfully.")


# --- Command-Line Interface ---
class CliEntryPoint:
    """
    CLI entry point for building configuration and running the evaluation.

    This class provides a command-line interface for running KVPress evaluations.
    Configuration can be specified via:
    1. YAML config file (default: "./evaluate_config.yaml")
    2. Command-line arguments (highest priority)
    """

    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        """
        Builds the configuration and runs the evaluation.

        Configuration is built by layering:
        1. Default values from EvaluationConfig
        2. Values from YAML config file
        3. Command-line arguments (highest priority)
        """
        # 1. Start with dataclass defaults.
        final_args = asdict(EvaluationConfig())

        # 2. Layer YAML values on top.
        yaml_config = _load_yaml_config(config_file)
        final_args.update(yaml_config)

        # 3. Layer CLI arguments on top (highest priority).
        # Filter out None values from CLI overrides
        cli_args = {k: v for k, v in cli_overrides.items() if v is not None}
        final_args.update(cli_args)

        # 4. Create and validate the final config object.
        try:
            config = EvaluationConfig(**final_args)
        except TypeError as e:
            # Provide a user-friendly error for bad arguments.
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)
