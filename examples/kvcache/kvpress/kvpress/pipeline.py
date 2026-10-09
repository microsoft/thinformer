# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import contextlib
import logging
from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Cache, DynamicCache, Pipeline, QuantizedCache
# Generation utils
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from kvpress.presses.base_press import BasePress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.finch_press import FinchPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.prefill_decoding_press import PrefillDecodingPress

logger = logging.getLogger(__name__)


class KVPressTextGenerationPipeline(Pipeline):
    """
    Pipeline for key-value cache compression in causal language models.

    Enables efficient processing of long contexts by applying KV cache compression
    during pre-filling, then generating answers using greedy decoding.

    Example:
    ```python
    pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    press = SnapKVPress(compression_ratio=0.5)
    result = pipeline(context="Long text...", question="A question about the long context.", press=press)
    ```
    """

    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        cache: Optional[Cache] = None,
        do_sample: bool = False,
        disable_chat_template: bool = False,
        query_events: Optional[dict] = None,
        prefill_context_events: Optional[dict] = None,
        prefill_question_events: Optional[list[dict]] = None,
        cache_sizes_tracking: Optional[dict] = None,
        **kwargs,
    ):
        """
        Sanitize the input parameters for the pipeline.
        The user can either provide a single question or a list of questions to be asked about the context.

        Parameters
        ----------
        question : str, optional
            The question to be asked about the context. Exclusive with `questions`.
        questions : list[str], optional
            A list of questions to be asked about the context. Exclusive with `question`.
        answer_prefix : str, optional
            The prefix to be added to the generated answer.
        press : BasePress, optional
            The key-value cache compression method to apply during pre-filling.

            Accepts any KVPress compression method (SnapKVPress, KnormPress,
            ExpectedAttentionPress, BlockPress, AdaKVPress, ComposedPress, etc.).
            If None, no compression is applied.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate for each answer.
        max_context_length : int, optional
            The maximum number of tokens in the context. By default will use the maximum length supported by the model.
        cache : Cache, optional
            The cache to use for the forward pass. Defaults to None (DynamicCache).
        do_sample : bool, default=False
            Whether to sample from the model's output distribution.
        disable_chat_template : bool, default=False
            If True, skip applying the tokenizer chat template (prepend the BOS token
            and a newline separator instead).
        query_events : dict, optional
            Dictionary to track query CUDA events for each layer.
        prefill_context_events : dict, optional
            Dictionary to track context prefill CUDA events for each layer.
            NOTE: the context is shared across questions, so only one dictionary is needed.
        prefill_question_events : list[dict], optional
            List of dictionaries to track question prefill CUDA events for each layer, for each question.
            The list is indexed by the question index.
        cache_sizes_tracking : dict, optional
            Dictionary to track cache sizes at each decoding step for each layer.
        **kwargs : dict
            Additional keyword arguments, currently ignored.

        Returns
        -------
        Tuple[dict, dict, dict]
            A tuple containing three dictionaries:
                - preprocess_kwargs: The keyword arguments for the preprocess function.
                - forward_kwargs: The keyword arguments for the forward function.
                - postprocess_kwargs: The keyword arguments for the postprocess function.
        """

        answer_prefix = answer_prefix or ""
        postprocess_kwargs = {"single_question": questions is None}
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        questions = questions or ([question] if question else [""])
        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))  # 1e10 to avoid overflow
        preprocess_kwargs = {
            "questions": questions,
            "answer_prefix": answer_prefix,
            "max_context_length": max_context_length,
            "disable_chat_template": disable_chat_template,
        }
        forward_kwargs = {"press": press, "max_new_tokens": max_new_tokens, "cache": cache, "do_sample": do_sample, "query_events": query_events, "prefill_context_events": prefill_context_events, "prefill_question_events": prefill_question_events, "cache_sizes_tracking": cache_sizes_tracking}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
        disable_chat_template: bool = False,
    ):
        """
        Apply chat template and tokenize the context and questions.

        Prepares input text for KV cache compression and generation by applying
        appropriate chat templates and tokenizing. Handles models with and without
        chat templates.

        Parameters
        ----------
        context : str
            Long context text to be compressed using the press method.
        questions : list[str]
            Questions to be asked about the context.
        answer_prefix : str
            Optional prefix for generated answers.
        max_context_length : int
            Maximum tokens allowed in context (truncated if exceeded).
        disable_chat_template : bool, default=False
            If True, skip applying the tokenizer chat template.

        Returns
        -------
        dict[str, GenericTensor]
            Dictionary with "context_ids" and "questions_ids" tensors.
        """

        # Apply chat template if available
        if disable_chat_template or self.tokenizer.chat_template is None:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context = bos_token + context
            question_suffix = "\n"  # to separate the question from the answer
        else:
            separator = "\n" + "#" * len(context)
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            context, question_suffix = context.split(separator)

        # Add question_suffix and answer prefix
        # e.g. for llama3.1, question_suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        questions = [question + question_suffix + answer_prefix for question in questions]

        # Tokenize the context and questions
        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) for question in questions
        ]

        # Truncate context
        if context_ids.shape[1] > max_context_length:
            logger.warning(
                f"Context length has been truncated from {context_ids.shape[1]} to {max_context_length} tokens."
            )
            context_ids = context_ids[:, :max_context_length]

        return {"context_ids": context_ids, "questions_ids": question_ids}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
        do_sample: bool = False,
        query_events: Optional[list[dict]] = None,
        prefill_context_events: Optional[dict] = None,
        prefill_question_events: Optional[list[dict]] = None,
        cache_sizes_tracking: Optional[dict] = None,
    ):
        """
        Execute KV cache compression and text generation pipeline.

        Performs context compression using the press method during pre-filling,
        then generates answers using greedy decoding or sampling.

        Parameters
        ----------
        input_tensors : dict[str, GenericTensor]
            Tokenized inputs with "context_ids" and "questions_ids".
        max_new_tokens : int, default=50
            Maximum tokens to generate for each answer.
        press : BasePress, optional
            Compression method for context pre-filling. If None, no compression.
        cache : Cache, optional
            Cache object for forward pass. If None, creates new DynamicCache.
        do_sample : bool, default=False
            Whether to sample from the model's output distribution.
        query_events : list[dict], optional
            List of dictionaries to track query CUDA events for each layer, for each question.
            The list is indexed by the question index.
        prefill_context_events : dict, optional
            Dictionary to track context prefill CUDA events for each layer.
            NOTE: the context is shared across questions, so only one dictionary is needed.
        prefill_question_events : list[dict], optional
            List of dictionaries to track question prefill CUDA events for each layer, for each question.
            The list is indexed by the question index.
        cache_sizes_tracking : dict, optional
            Dictionary to track cache sizes at each decoding step for each layer.

        Returns
        -------
        list[str]
            Generated answers for each input question.
        """
        if isinstance(press, (DecodingPress, PrefillDecodingPress)) and len(input_tensors["questions_ids"]) > 1:
            raise ValueError(
                "DecodingPress is not compatible with multiple questions. Please specify a single question."
            )

        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        # Prefilling using the press on the context
        if cache is None:
            cache = DynamicCache()

        # We only perform prefill compression if the press is a prefill press
        perform_prefill_compression = press is not None and not isinstance(press, DecodingPress)
        with press(self.model) if perform_prefill_compression else contextlib.nullcontext():
            # We run the model without the lm head for pre-filling.
            self.model.model(
                input_ids=context_ids,
                past_key_values=cache,
                output_attentions=self.output_attentions(press),
                query_events=prefill_context_events,
            )

            logger.debug(f"Context Length: {context_length}")
            logger.debug(f"Compressed Context Length: {cache.get_seq_length()}")

        # We only perform decoding compression if the press is a decoding or prefill decoding press
        perform_decoding_compression = press is not None and isinstance(press, (DecodingPress, PrefillDecodingPress))
        with press(self.model) if perform_decoding_compression else contextlib.nullcontext():
            # Greedy decoding for each question
            answers = []
            for i, question_ids in enumerate(input_tensors["questions_ids"]):
                if isinstance(press, KeyRerotationPress) or (isinstance(press, FinchPress) and press.rerotate_keys):
                    context_length = cache.get_seq_length()

                cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
                answer, completion_tokens = self.generate_answer(
                    question_ids=question_ids.to(self.model.device),
                    cache=cache,
                    context_length=context_length,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    query_events=query_events[i] if query_events is not None else None,
                    prefill_question_events=prefill_question_events[i] if prefill_question_events is not None else None,
                    cache_sizes_tracking=cache_sizes_tracking,
                )
                self._remove_answer_from_cache(cache, cache_seq_lengths)
                prompt_tokens = context_length + question_ids.shape[1]
                answers.append((answer, prompt_tokens, completion_tokens))
        return answers

    def _remove_answer_from_cache(self, cache: Cache, cache_seq_lengths: list[int]):

        for layer_idx, sequence_length in enumerate(cache_seq_lengths):
            cache.layers[layer_idx].keys = cache.layers[layer_idx].keys[:, :, :sequence_length]
            cache.layers[layer_idx].values = cache.layers[layer_idx].values[:, :, :sequence_length]

        if isinstance(cache, QuantizedCache):
            for layer_idx, sequence_length in enumerate(cache_seq_lengths):
                cache.layers[layer_idx]._quantized_keys = cache.layers[layer_idx]._quantized_keys[
                    :, :, :sequence_length
                ]
                cache.layers[layer_idx]._quantized_values = cache.layers[layer_idx]._quantized_values[
                    :, :, :sequence_length
                ]

    def generate_answer(
        self, question_ids: torch.Tensor, cache: Cache, context_length: int, max_new_tokens: int, do_sample: bool = False, query_events: Optional[dict] = None, prefill_question_events: Optional[dict] = None, cache_sizes_tracking: Optional[dict] = None
    ) -> tuple[str, int]:
        """
        Generate an answer to a question using greedy decoding or sampling.

        Parameters
        ----------
        question_ids : torch.Tensor
            The tokenized question.
        cache : Cache
            The compressed key-value cache.
        context_length : int
            The length of the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.
        do_sample : bool, default=False
            Whether to sample from the model's output distribution.
        query_events : Optional[dict], default=None
            Dictionary to track query CUDA events for each layer.
        prefill_question_events : Optional[dict], default=None
            Dictionary to track question prefill CUDA events for each layer (for the question forward pass).
        cache_sizes_tracking : Optional[dict], default=None
            Dictionary to track cache sizes at each decoding step for each layer.

        Returns
        -------
        str
            The generated answer.
        int
            The number of tokens generated.
        """
        position_ids = torch.arange(
            context_length, context_length + question_ids.shape[1], device=self.model.device
        ).unsqueeze(0)

        # if the user doesn't provide a question, skip forward pass
        outputs = self.model(
            input_ids=question_ids.to(self.model.device),
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
            query_events=prefill_question_events,
        )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]
        for i in range(max_new_tokens - 1):
            input_ids = generated_ids[-1].unsqueeze(0).unsqueeze(0)
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                position_ids=position_ids + i,
                query_events=query_events,
            )
            
            # Record cache sizes for each layer at this decoding step
            if cache_sizes_tracking is not None:
                for layer_idx in range(len(cache)):
                    layer_key = str(layer_idx)  # Convert to string for Parquet compatibility
                    if layer_key not in cache_sizes_tracking:
                        cache_sizes_tracking[layer_key] = []
                    cache_sizes_tracking[layer_key].append(cache.layers[layer_idx].keys.shape[2])
            
            # From https://github.com/huggingface/transformers/blob/77e8b9f8dfc8e736ad2f603a5b2ae2b1076ed271/src/transformers/generation/utils.py#L2916
            if do_sample:
                # Create list of logit processors
                # Ref: https://github.com/huggingface/transformers/blob/77e8b9f8dfc8e736ad2f603a5b2ae2b1076ed271/src/transformers/generation/utils.py#L1292
                logits_processors = LogitsProcessorList()
                generation_config = self.model.generation_config
                if generation_config.temperature is not None and generation_config.temperature != 1.0:
                    logits_processors.append(TemperatureLogitsWarper(generation_config.temperature))
                if generation_config.top_p is not None and generation_config.top_p < 1.0:
                    logits_processors.append(TopPLogitsWarper(top_p=generation_config.top_p))
                # import pdb; pdb.set_trace()
                next_token_logits = outputs.logits[:, -1]
                # NOTE: for the temperature and top_p logits warpers, input_ids is ignored, 
                # but we need to pass it to the logits processors interface anyways
                next_token_scores = logits_processors(input_ids, next_token_logits)
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                new_id = torch.multinomial(probs, num_samples=1).squeeze(0).squeeze(0)
            else:
                new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break
        answer = self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)
        return answer, len(generated_ids)

    def output_attentions(self, press: BasePress):
        if isinstance(press, ObservedAttentionPress):
            return True
        if hasattr(press, "press") and isinstance(press.press, ObservedAttentionPress):
            return True
        return False

    def postprocess(self, model_outputs, single_question):
        if single_question:
            return {"answer": model_outputs[0]}
        return {"answers": model_outputs}


PIPELINE_REGISTRY.register_pipeline(
    "kv-press-text-generation",
    pipeline_class=KVPressTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
