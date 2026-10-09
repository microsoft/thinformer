# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.aime25.calculate_metrics import calculate_metrics as aime25_scorer
from benchmarks.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics_e as longbench_scorer_e
from benchmarks.longbenchv2.calculate_metrics import calculate_metrics as longbenchv2_scorer
from benchmarks.loogle.calculate_metrics import calculate_metrics as loogle_scorer
from benchmarks.math500.calculate_metrics import calculate_metrics as math500_scorer
from benchmarks.needle_in_haystack.calculate_metrics import calculate_metrics as needle_in_haystack_scorer
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from benchmarks.zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer

from kvpress import (
    AdaKVPress,
    BlockPress,
    ChunkKVPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVzipPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.express_new import ExpressNew
# from kvpress.presses.express_prefill import Express as ExpressPrefill
from kvpress.presses.balancekv_press import BalanceKVPress, BalanceKVScorerPress

# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
# Each entry is a (path, revision) tuple; revision=None loads the latest version.
DATASET_REGISTRY = {
    "loogle": ("simonjegou/loogle", None),
    "ruler": ("simonjegou/ruler", None),
    "zero_scrolls": ("simonjegou/zero_scrolls", None),
    "infinitebench": ("MaxJeblick/InfiniteBench", None),
    "longbench": ("Xnhyacinth/LongBench", None),
    "longbench-e": ("Xnhyacinth/LongBench", None),
    "longbench-v2": ("Xnhyacinth/LongBench-v2", None),
    "needle_in_haystack": ("alessiodevoto/paul_graham_essays", None),
    # Datasets used to be used for decoding compression
    "aime25": ("alessiodevoto/aime25", None),
    "math500": ("alessiodevoto/math500", "5c06bb9ee02a115fff716c0a1be1a18b1005e63e"),
}

SCORER_REGISTRY = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
    "longbench-e": longbench_scorer_e,
    "longbench-v2": longbenchv2_scorer,
    "needle_in_haystack": needle_in_haystack_scorer,
    "aime25": aime25_scorer,
    "math500": math500_scorer,
}


PRESS_REGISTRY = {
    "adakv_expected_attention": AdaKVPress(ExpectedAttentionPress()),
    "adakv_expected_attention_e2": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "adakv_snapkv": AdaKVPress(SnapKVPress()),
    "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "critical_adakv_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
    "critical_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_snapkv": CriticalKVPress(SnapKVPress()),
    "duo_attention": DuoAttentionPress(),
    "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
    "expected_attention": ExpectedAttentionPress(),
    "finch": FinchPress(),
    "keydiff": KeyDiffPress(),
    "kvzip": KVzipPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "pyramidkv": PyramidKVPress(),
    "qfilter": QFilterPress(),
    "random": RandomPress(),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "no_press": None,
    "decoding_knorm": DecodingPress(base_press=KnormPress()),
    "decoding_streaming_llm": DecodingPress(base_press=StreamingLLMPress(), compression_interval=1),
    "decoding_tova": DecodingPress(base_press=TOVAPress()),
    "decoding_qfilter": DecodingPress(base_press=QFilterPress()),
    "decoding_adakv_expected_attention_e2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2))),
    "decoding_adakv_snapkv": DecodingPress(base_press=AdaKVPress(SnapKVPress())),
    "decoding_keydiff": DecodingPress(base_press=KeyDiffPress()),
    "decoding_snapkv": DecodingPress(base_press=SnapKVPress()),
    "decoding_expected_attention": DecodingPress(base_press=ExpectedAttentionPress(epsilon=1e-2)),
    "express_new_32_32": ExpressNew(base_press=KnormPress(), n_sink=32, n_window=32),
    "balancekv_express": ExpressNew(base_press=KnormPress(), n_sink=32, n_window=32, use_balancekv=True),
    "balancekv": BalanceKVPress(),
    "balancekv_scorer": BalanceKVScorerPress(),
    "decoding_balancekv": DecodingPress(base_press=BalanceKVScorerPress(), size_based_trigger=True),
}
