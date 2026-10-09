"""Script for measuring the runtime of a single attention layer.

Adapted from https://github.com/amirzandieh/HyperAttention/blob/main/benchmark_single_attention.py

Example Usage:
```bash
python benchmark_single_attention.py -m METHOD -bs 1 -hs 32 -d 128 -s 1 -op /path/to/output/dir
```
Settings:
- In-network: timing the attention layer inside the chatglm2-6b-32k model.
- Out-of-network: timing the attention layer as a standalone module.
In both settings, we use random input tensors of the appropriate shape.
"""

import torch
import os
import pandas as pd
import logging
from transformers import AutoModelForCausalLM
from collections import defaultdict
import numpy as np
from slugify import slugify

from thinformer.triton import (
    ThinformerAttention as ThinformerAttentionTriton,
)
from thinformer.thinformer import ThinformerAttention
from util_experiments import get_base_parser, get_tensors, get_tensors_from_vocab
from models.replace_llm_attention import patch_attention_layers

from models.attention.hyper_attn import HyperAttention
from thinformer.timing_utils import multi_events
from thinformer.memory_utils import profiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_attn_config(attn_method: str, h: int) -> dict:
    """Get the configuration for the attention method.

    Args:
        attn_method (str): attention method to benchmark
        h (int): log2 of target cache size for express attention
        M_inner (int): inner dimension for express_triton attention

    Returns:
        cfg (dict): configuration for the attention method

    """
    match attn_method:
        case "exact":
            cfg = {}
        case "hyper-cuda":
            cfg = {
                "block_size": 256,
                "sample_size": 256,
                "lsh_num_projs": 7,
                "min_seq_len": 4096,
                "cuda": True,  # "cuda" in attn_method
            }
        case "thinformer":
            cfg = {
                "use_torch_spda": True,
                "log2_target_cache_size": h,
            }
        case "thinformer_triton":
            cfg = {
                "log2_target_cache_size": h,
            }
        case _:
            raise ValueError(f"Invalid attention method: {attn_method}")
    return cfg


def get_avg_attn_time(times: dict[str, list[float]]) -> float:
    """Get the average attention time from the timing results.

    Args:
        times (dict): dictionary of times for each event

    Returns:
        avg_time (float): average attention time

    """
    attn_times = [np.mean(v) for k, v in times.items() if k.endswith("_attn")]
    return np.mean(attn_times) if attn_times else 0.0


def run_benchmark_in_network(
    attn_method: str,
    batch_size: int,
    head_size: int,
    seq_len: int,
    dim: int,
    mode: str,
    h: int = 10,
    warmup: int = 20,
    rep: int = 100,
    num_patch_layers: int = 0,
    model_name: str = "THUDM/chatglm2-6b-32k",
    causal: bool = True,
    n_first: int = 0,
    n_last: int = 0,
) -> dict[str, list[float]]:
    """Run the benchmark for a single attention layer.

    Args:
        attn_method (str): attention method to benchmark
        batch_size (int): batch size
        head_size (int): head size
        seq_len (int): sequence length
        dim (int): dimension
        mode (str): mode to run the benchmark
        warmup (int): number of warm-up runs
        rep (int): number of runs (excluding warm-up runs)
        h (int): log2 of target cache size for Thinformer
        num_patch_layers (int): number of layers to patch in the model
        M_inner (int): block size for kernel halving
        model_name: HuggingFace repo name
        causal (bool): whether to use causal attention
        n_first (int): number of attention sinks
        n_last (int): size of sliding window

    Returns:
        times (dict): dictionary of times for each event

    """
    match model_name:
        case "THUDM/chatglm2-6b-32k":
            model = AutoModelForCausalLM.from_pretrained(
                "THUDM/chatglm2-6b-32k",
                trust_remote_code=True,
                empty_init=False,
                torch_dtype=torch.float32,
            )
        case "meta-llama/Meta-Llama-3.1-8B-Instruct":
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B-Instruct"
            )
    patch_layers = -1
    if num_patch_layers > 0:
        patch_layers = num_patch_layers
    # we only replace the second layer of the model for non-causal attention
    # as torch.compile in "thinformer" fails when applied to every layer of the model
    patch_attention_layers(
        model=model,
        model_name=model_name,
        patch_config="last" if causal else "second",
        num_patch_layers=patch_layers,  # patch all layers
        attn_method=attn_method,
        causal=causal,
        n_first=n_first,
        n_last=n_last,
        **get_attn_config(attn_method, h=h),
    )
    match model_name:
        case "THUDM/chatglm2-6b-32k":
            model.to(device="cuda", dtype=torch.float16)
        case "meta-llama/Meta-Llama-3.1-8B-Instruct":
            model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    # get input ids with shape (batch_size, seq_len)
    x = torch.randint(
        low=0,
        high=65024,
        size=(batch_size, seq_len),
        device="cuda",
        dtype=torch.int64,
    )

    def fn() -> torch.Tensor:
        return model(x)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        # Run benchmark
        times = defaultdict(list)
        usage = defaultdict(list)
        for ii in range(rep):
            # Reset profiling object
            multi_events.reset()
            profiler.reset()
            # print(f'on iteration {ii}')
            # Run model forward pass to collect timings
            with multi_events("all"):
                _ = fn()
            # Ensure all GPU computation has completed
            torch.cuda.synchronize()
            # Calculate runtimes for events inside the Thinformer module
            # import pdb; pdb.set_trace()
            for event_name, events_list in multi_events.items():
                times[event_name].append(
                    [start.elapsed_time(end) for start, end in events_list]
                )
            for name, amt in profiler.items():
                # Use the formula provided by
                # https://github.com/Dao-AILab/flash-attention/blob/7321879fde54f09ed94f7f6ce9377e2f4cf1fac0/flash_attn/utils/benchmark.py#L258
                usage[name].append(amt / ((2**20) * 1000))
        # aggregate times per run
        for event_name in times.keys():
            times[event_name] = [sum(x) for x in times[event_name]]
    # return 0.5, 0.2, 0.8 quantiles of attn time
    # return np.quantile(times["attn"], [0.5, 0.2, 0.8])
    return times, usage


def run_benchmark_out_of_network(
    attn_method: str,
    batch_size: int,
    head_size: int,
    seq_len: int,
    dim: int,
    mode: str,
    h: int = 10,
    warmup: int = 20,
    rep: int = 100,
    num_patch_layers: int = 0,
    causal: bool = True,
    n_first: int = 0,
    n_last: int = 0,
    input_source: str = "random",
    model_name: str = "THUDM/chatglm2-6b-32k",
    save_outputs: bool = False,
    **kwargs: dict,
) -> dict[str, list[float]]:
    """Run the benchmark for a single attention layer.

    Args:
        attn_method (str): attention method to benchmark
        batch_size (int): batch size
        head_size (int): head size
        seq_len (int): sequence length
        dim (int): dimension
        mode (str): mode to run the benchmark
        warmup (int): number of warm-up runs
        rep (int): number of runs (excluding warm-up runs)
        h (int): log2 of target cache size for Thinformer
        num_patch_layers (int): number of layers to patch in the model
        M_inner (int): block size for kernel halving
        causal (bool): whether to use causal attention
        n_first (int): number of attention sinks
        n_last (int): size of sliding window
        input_source (str): 'random' Gaussian noise or 'glm_embed' seeded from
            sampled word embeddings of `model_name`
        model_name (str): model whose embeddings seed 'glm_embed' inputs
        save_outputs (bool): dump the attention output of the last timed run
            to Out_data_sweep[_<input_source>]/ for compute_maxnorm_truncate.py
        kwargs (dict): additional keyword args

    Returns:
        times (dict): dictionary of times for each event

    """
    if input_source == "glm_embed":
        q, k, v = get_tensors_from_vocab(
            batch_size, seq_len, head_size, dim, model_name=model_name
        )
    else:
        q, k, v = get_tensors(batch_size, seq_len, head_size, dim)
    sm_scale = 1.0 / (dim**0.5)

    match attn_method:
        case "exact":

            def forward_fn() -> torch.Tensor:
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=causal,
                    dropout_p=0.0,
                    scale=sm_scale,
                )
        case "thinformer_triton":
            attn = ThinformerAttentionTriton(
                scale=sm_scale, h=h, alias="1_thinformer"
            )

            def forward_fn() -> torch.Tensor:
                return attn(q, k, v, causal)
        case "thinformer":
            attn = ThinformerAttention(
                scale=sm_scale,
                use_torch_spda=True,
                h=h,
            )
            assert causal is False, "Thinformer does not support causal attention"
            def forward_fn() -> torch.Tensor:
                return attn(q, k, v)

        case "hyper-cuda":
            # NOTE: always use hyper-cuda since the triton version is really slow
            block_size = 256
            sample_size = 256
            cuda = True

            attn = HyperAttention(
                input_dim=dim,
                block_size=block_size,
                sample_size=sample_size,
                min_seq_len=4096,
                cuda=cuda,
                n_first=n_first,
                n_last=n_last,
            ).to(device="cuda", dtype=q.dtype)

            def forward_fn() -> torch.Tensor:
                return attn(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    causal=causal,
                )
        case _:
            raise NotImplementedError(
                f"Attention method '{attn_method}' not implemented."
            )

    # Warm-up runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_fn()
        # Run benchmark
        times = defaultdict(list)
        usage = defaultdict(list)
        for ii in range(rep):
            # Reset events object
            multi_events.reset()
            profiler.reset()
            # Run model forward pass to collect timings
            with multi_events("1_attn"):
                out = forward_fn()
            # Ensure all GPU computation has completed
            torch.cuda.synchronize()
            # Calculate runtimes for events inside the Thinformer module
            # import pdb; pdb.set_trace()
            for event_name, events_list in multi_events.items():
                times[event_name].append(
                    [start.elapsed_time(end) for start, end in events_list]
                )
            for name, amt in profiler.items():
                usage[name].append(amt / ((2**20) * 1000))
        # aggregate times per run
        for event_name in times.keys():
            times[event_name] = [sum(x) for x in times[event_name]]
        if save_outputs:
            save_attention_output(
                out, attn_method, h, seq_len, input_source
            )
    return times, usage


def save_attention_output(
    out: torch.Tensor | tuple,
    attn_method: str,
    h: int,
    seq_len: int,
    input_source: str,
) -> str:
    """Dump one attention output as float16 (B, N, H, D) for the max-norm
    error analysis (compute_maxnorm_truncate.py).

    Files go to Out_data_sweep/ (random inputs) or Out_data_sweep_<input_source>/
    as Out_layer_<method>[_h<h>]_seq<seq_len>.npz with keys "out" and
    "layout" (= "bnhd"). Only Thinformer outputs depend on h, so exact and the
    other methods get no h tag and are shared across h.
    """
    if isinstance(out, tuple):  # ThinformerAttentionTriton returns (out, None)
        out = out[0]
    if attn_method in ("exact", "hyper-cuda"):
        # these kernels take/return (B, H, N, D); bring to the (B, N, H, D) layout
        out = out.transpose(1, 2)
    out_dir = (
        "Out_data_sweep" if input_source == "random" else f"Out_data_sweep_{input_source}"
    )
    os.makedirs(out_dir, exist_ok=True)
    h_tag = f"_h{h}" if attn_method.startswith("thinformer") else ""
    path = os.path.join(out_dir, f"Out_layer_{attn_method}{h_tag}_seq{seq_len}.npz")
    np.savez(
        path, out=out.detach().contiguous().to(torch.float16).cpu().numpy(), layout="bnhd"
    )
    print(f"Attention output saved to {path}")
    return path


parser = get_base_parser()
parser.add_argument(
    "--in_network",
    action="store_true",
    help="time the attention layer in the chatglm2-6b-32k model",
)
parser.add_argument(
    "--time_all",
    action="store_true",
    help="time all layers in the model",
)
parser.add_argument(
    "--num_patch_layers",
    type=int,
    default=-1,
    help="number of patch layers in the model",
)
parser.add_argument(
    "--seq_len",
    type=int,
    default=0,
    help="sequence length for the model",  # runs on all seq len if not set
)
parser.add_argument(
    "--num_warmup",
    type=int,
    default=20,
    help="number of warmup runs before timing the model",
)
parser.add_argument(
    "--save_outputs",
    action="store_true",
    help="(out-of-network only) dump the attention output of the last timed "
    "run to Out_data_sweep[_<input_source>]/ for compute_maxnorm_truncate.py",
)
args = parser.parse_args()

if args.time_all and not args.in_network:
    raise ValueError("Can only time all layers when --in_network is set.")
if args.save_outputs and args.in_network:
    raise ValueError("--save_outputs is only supported out-of-network.")

for arg_name, arg_var in args.__dict__.items():
    print(f"{arg_name:<16} : {arg_var}")

torch._dynamo.config.cache_size_limit = 32

attn_method = args.method
mode = args.mode
seed = args.seed
force = args.force
batch_size = args.batch_size
head_size = args.head_size
dim = args.dim
h = args.log2_target_cache_size
time_all = args.time_all
num_patch_layers = args.num_patch_layers
warmup = args.num_warmup
in_network = args.in_network
model_name = args.model_name
model_slug = slugify(model_name)
causal = bool(args.causal)
n_first = args.n_first
n_last = args.n_last
input_source = args.input_source

if in_network:
    run_benchmark = run_benchmark_in_network
    # in-network always uses real token ids; the input_source flag and output
    # dumps are out-of-network only.
    extra_kwargs = {}
else:
    run_benchmark = run_benchmark_out_of_network
    extra_kwargs = {"input_source": input_source, "save_outputs": args.save_outputs}
    # enable the memory profiler
    profiler.enable()

if args.seq_len > 0:
    seq_lens = [args.seq_len]
else:
    if in_network:
        seq_lens = [
            2**i for i in range(10, 16)
        ]  # up to 32k (since we are limited by chatglm2-6b-32k)
    else:
        if attn_method == "hyper-cuda":
            logger.info("Using smaller sequence lengths for hyper-cuda to avoid OOM")
            seq_lens = [2**i for i in range(10, 19)]  # up to 256k
        else:
            seq_lens = [2**i for i in range(10, 20)]  # up to 512k

rep = args.num_runs
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)
times_dir = os.path.join(output_path, "times")
os.makedirs(times_dir, exist_ok=True)
# Include seq_len in the filename when a single sequence length is requested, so that
# per-seq_len runs can co-exist in one output dir. When --seq_len is omitted, the script
# loops over all seq_lens into one combined CSV, so the suffix is empty (filename unchanged).
seq_suffix = f"-seq{args.seq_len}" if args.seq_len > 0 else ""
filename = os.path.join(
    times_dir,
    f"times-n{rep}-{attn_method}-{mode}-bs{batch_size}-h{head_size}-d{dim}-s{seed}-h{h}-{in_network}-causal{causal}-{model_slug}{seq_suffix}.csv",
)
if os.path.exists(filename) and not force and not time_all:
    print(f"Skipping {filename} since it already exists. Use --force to overwrite.")
    exit()

results = []
if time_all:
    results_all = []

for seq_len in seq_lens:
    # set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # globally disable sdpa kernel other than flash attention
    # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    times, usage = run_benchmark(
        attn_method,
        batch_size,
        head_size,
        seq_len,
        dim,
        mode,
        rep=rep,
        h=h,
        num_patch_layers=num_patch_layers,
        warmup=warmup,
        model_name=model_name,
        causal=causal,
        n_first=n_first,
        n_last=n_last,
        **extra_kwargs,
    )
    # print(f'times: {times.keys()}')
    # print(f'time attn ')
    if "1_attn" in times:
        ms = np.quantile(times["1_attn"], [0.5, 0.2, 0.8])
        print(
            f"[{mode:<8}], {attn_method}, seq_len: {seq_len:<8}, seed: {seed}, ms: {ms[0]:.5f} ({ms[1]:.5f}, {ms[2]:.5f}) | "
        )
    else:
        print(
            f"[{mode:<8}], {attn_method}, seq_len: {seq_len:<8}, seed: {seed}, ms: None | "
        )
    df = pd.DataFrame(
        data={
            "method": attn_method,
            "model": model_name,
            "mode": mode,
            "seq_len": seq_len,
            "h": h,
            "in_network": in_network,
            **{
                k: v for k, v in times.items() if k.startswith("1_")
            },  # only keep timings from the first attention layer
            # **{f"mem_{k}": v for k, v in usage.items() if k.startswith("1_")},
        }
    )
    results.append(df)
    if time_all:
        row = {
            "method": attn_method,
            "model": model_name,
            "mode": mode,
            "seq_len": seq_len,
            "num_patch_layers": num_patch_layers,
            **{k: float(np.mean(v)) for k, v in times.items() if k.endswith("_attn")},
            # **{
            #     f"mem_{k}": float(np.mean(v))
            #     for k, v in usage.items()
            #     if k.endswith("_attn")
            # },
        }
        df = pd.DataFrame([row])
        df["all_attn"] = get_avg_attn_time(times)
        results_all.append(df)

df = pd.concat(results)
df.to_csv(filename, index=False)

print(f"\nResults saved to {filename}")

if time_all:
    times_all_dir = os.path.join(output_path, "times_all")
    os.makedirs(times_all_dir, exist_ok=True)
    filename = f"times_all-n{rep}-{attn_method}-{mode}-bs{batch_size}-h{head_size}-d{dim}-s{seed}-h{h}-{in_network}-causal{causal}-{model_slug}{seq_suffix}.csv"
    df_all = pd.concat(results_all)
    os.makedirs(times_all_dir, exist_ok=True)
    df_all.to_csv(os.path.join(times_all_dir, filename), index=False)
    print(f"Results (all layers) saved to {os.path.join(times_all_dir, filename)}")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
df_profile = df.groupby(["method", "mode", "seq_len", "h"])[numeric_cols].quantile(
    [0.2, 0.5, 0.8]
)
profiles_dir = os.path.join(output_path, "profiles")
os.makedirs(profiles_dir, exist_ok=True)
save_path = os.path.join(
    profiles_dir,
    f"profile-n{rep}-{attn_method}-{mode}-bs{batch_size}-h{head_size}-d{dim}-s{seed}-h{h}-{in_network}-causal{causal}-{model_slug}{seq_suffix}.csv",
)
df_profile.to_csv(save_path)

print(f"Profile results saved to {save_path}")
