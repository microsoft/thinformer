import argparse
import torch
import random
import numpy as np


def get_base_parser() -> argparse.ArgumentParser:
    """Get the base parser for the longbench attention experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="fwd",
        choices=["fwd", "bwd", "fwd+bwd"],
        help="Benchmarking mode.",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="exact",
        choices=[
            "exact",
            "thinformer_triton",
            "thinformer",
            "hyper-cuda",
        ],
        help="Attention method to benchmark.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=1,
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_path",
        "-op",
        type=str,
        default="out",
        help="Directory for storing output.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="if set, overwrite existing output file even when it exists",
    )
    parser.add_argument(
        "--num_runs",
        "-n",
        default=100,
        type=int,
        help="number of runs (excluding warm-up runs)",
    )
    parser.add_argument(
        "--head_size",
        "-hs",
        type=int,
        default=32,
        help="number of attention heads",
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=64,
        help="per-head dimension",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--log2_target_cache_size",
        type=int,
        help="log2 of target cache size (nout) for Thinformer Express",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="THUDM/chatglm2-6b-32k",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--n_first",
        type=int,
        default=0,
        help="number of attention sinks",
    )
    parser.add_argument(
        "--n_last",
        type=int,
        default=0,
        help="size of sliding window",
    )
    parser.add_argument(
        "--causal",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to use causal attention (1) or not (0)",
    )
    parser.add_argument(
        "--input_source",
        type=str,
        default="random",
        choices=["random", "glm_embed"],
        help="how to generate out-of-network q,k,v: 'random' Gaussian noise, "
        "or 'glm_embed' seeded from sampled word embeddings of --model_name",
    )
    return parser


def get_tensors(
    batch_size: int, seq_len: int, head_size: int, dim: int, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random input tensors for a single attention layer.

    Args:
        batch_size: int
        seq_len: int
        head_size: int
        dim: int
        device: str

    Returns:
        q (torch.Tensor): query tensor of shape (batch_size, seq_len, head_size, dim)
        k (torch.Tensor): key tensor of shape (batch_size, seq_len, head_size, dim)
        v (torch.Tensor): value tensor of shape (batch_size, seq_len, head_size, dim)

    """
    q = torch.randn(
        (batch_size, seq_len, head_size, dim),
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    k = torch.randn(
        (batch_size, seq_len, head_size, dim),
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    v = torch.randn(
        (batch_size, seq_len, head_size, dim),
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    return q, k, v


_EMBED_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _get_glm_embeddings(model_name: str, device: str) -> torch.Tensor:
    """Load (and cache) the input word-embedding matrix of `model_name`.

    Returns the (vocab_size, hidden_size) weight tensor on `device`. The full
    model is loaded once to extract the embeddings, then freed.
    """
    key = (model_name, device)
    if key not in _EMBED_CACHE:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        weight = model.get_input_embeddings().weight.detach().to(device)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        _EMBED_CACHE[key] = weight
    return _EMBED_CACHE[key]


def get_tensors_from_vocab(
    batch_size: int,
    seq_len: int,
    head_size: int,
    dim: int,
    device: str = "cuda",
    model_name: str = "THUDM/chatglm2-6b-32k",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate input tensors seeded from real word embeddings.

    Instead of Gaussian noise, sample `seq_len` token ids uniformly from the
    model's vocabulary, look up their input embeddings (hidden_size == head_size
    * dim), and reshape into the (batch_size, seq_len, head_size, dim) attention
    layout. q, k, and v are drawn from three independent token samples so they
    are distinct (as they would be after the model's q/k/v projections).

    Args:
        batch_size: int
        seq_len: int
        head_size: int
        dim: int
        device: str
        model_name: HuggingFace repo whose embeddings seed the tensors

    Returns:
        q, k, v (torch.Tensor): each of shape (batch_size, seq_len, head_size, dim)

    """
    weight = _get_glm_embeddings(model_name, device)  # (vocab_size, hidden_size)
    vocab_size, hidden_size = weight.shape
    assert hidden_size == head_size * dim, (
        f"embedding hidden_size {hidden_size} != head_size*dim "
        f"{head_size * dim}; cannot reshape embeddings into (head_size, dim)"
    )

    def sample() -> torch.Tensor:
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        emb = weight[ids]  # (batch_size, seq_len, hidden_size)
        emb = emb.reshape(batch_size, seq_len, head_size, dim)
        return emb.detach().clone().requires_grad_(True)

    return sample(), sample(), sample()


def seed_everything(seed: int) -> None:
    """Helper function to set random seeds for reproducibility.

    Args:
        seed (int): Random seed for reproducibility.

    """
    # Python's built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNG for CPU
    torch.manual_seed(seed)

    # PyTorch RNG for CUDA (single and multi-GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # In PyTorch 2.6, also set the new RNG state for MPS (Apple Silicon) if relevant
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
