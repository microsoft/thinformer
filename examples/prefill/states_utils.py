"""Shared loader for the per-sample Q/K/V dumps written by benchmark_patch_llm.py.

benchmark_patch_llm.py --record_qkv_layer L writes, for every sequence,
    <output_dir>/states/states__model=<slug>_layer=<L>_..._bid=<i>.pt
holding a dict with
    query_states: [1, H,    n, d]
    key_states:   [1, Hkv,  n, d]
    value_states: [1, Hkv,  n, d]
    scaling:      float   softmax temperature s used by the layer (newer dumps)
(bf16, post q/k-norm and post-RoPE, exactly as passed to the attention kernel).

From each file we derive per-position curves that the plot_*.py scripts
aggregate over (sequence, head):
    Rq      (H,   n)  cummax_{i<=n} ||q_i||_2 per query head
    Rk      (H,   n)  r_n = cummax_{i<=n} ||k_i||_2 per query head. With
                      grouped-query attention (Hkv < H) the key head serving
                      each query head is repeated so Rk lines up with Rq
                      (repeat_kv order).
    q_norm  (H,   n)  ||q_n||_2 (per token, no cummax)
    log_Dnn (H,   n)  log D_nn,  D_nn = sum_{i<=n} exp(s <q_n, k_i>)
                      i.e. the causal softmax denominator of position n
    vmax    (Hkv, n)  cummax_{i<=n} ||v_i||_inf per key/value head
and from those, on the fly,
    log_rho (H,   n)  log rho_n,  rho_n = n / D_nn * exp(s ||q_n||^2 / 2) * exp(s r_n^2 / 2)

The temperature s defaults to the layer's own scaling as recorded in the dump
(1/sqrt(d) for Chameleon), falling back to 1/sqrt(d) for dumps without it.

Optional centering (matching the chatglm FastCoreAttention dumps on the
thinformer-express branch):
    center_keys    subtract kbar, the per-head mean key over the whole
                   sequence (non-causal), before Rk and D_nn. Softmax weights
                   are invariant to this, but D_nn changes by exp(-s <q_n, kbar>)
                   and r_n typically shrinks. Queries are never centered.
    center_values  subtract the per-head, per-coordinate midrange
                   (max + min) / 2 over the sequence before the inf-norm in vmax.

Loading a 32k states file is ~400 MB of I/O and D_nn needs the full causal
score matrix, so the derived arrays are cached as
<states_dir>/derived/<basename>[__scaling=<s>][__ckeys][__cvals].npz (a few
MB) and reused on later runs; a cache with missing keys is recomputed.
"""
import glob
import os
import re

import numpy as np
import torch

DERIVED_KEYS = ("Rq", "Rk", "q_norm", "log_Dnn", "vmax")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_ROW_CHUNK = 2048  # query rows per score block when computing D_nn


def find_state_files(data_dir, pattern="states__*.pt"):
    """Return the states files under data_dir matching pattern, sorted by bid."""

    def bid(p):
        m = re.search(r"bid=(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    files = sorted(glob.glob(os.path.join(data_dir, pattern)), key=bid)
    if not files:
        # The (large) .pt dumps may have been deleted after their derived caches
        # were built; fall back to the cache names so cached quantities still plot.
        cached = glob.glob(os.path.join(data_dir, "derived", pattern + "*.npz"))
        names = {os.path.basename(c).split(".pt")[0] + ".pt" for c in cached}
        files = sorted((os.path.join(data_dir, n) for n in names), key=bid)
    return files


def default_scaling(st, path):
    """The softmax temperature the layer actually used."""
    if "scaling" in st:
        return float(st["scaling"])
    return 1.0 / float(st["query_states"].shape[-1]) ** 0.5


def _causal_log_denominator(q, k, scaling):
    """log D_nn for every position n of one head: q, k are (n, d) float32."""
    n = q.shape[0]
    q = q.to(_DEVICE)
    k = k.to(_DEVICE)
    out = torch.empty(n, dtype=torch.float32)
    for start in range(0, n, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, n)
        scores = (q[start:stop] @ k[:stop].T) * scaling  # (rows, stop)
        rows = torch.arange(start, stop, device=scores.device)[:, None]
        cols = torch.arange(stop, device=scores.device)[None, :]
        scores.masked_fill_(cols > rows, float("-inf"))
        out[start:stop] = torch.logsumexp(scores, dim=-1).cpu()
        del scores
    return out


def _derive(path, scaling=None, center_keys=False, center_values=False):
    st = torch.load(path, map_location="cpu")
    if scaling is None:
        scaling = default_scaling(st, path)
    q = st["query_states"][0].float()  # (H, n, d)
    k = st["key_states"][0].float()  # (Hkv, n, d)
    v = st["value_states"][0].float()  # (Hkv, n, d)
    H, Hkv = q.shape[0], k.shape[0]
    assert H % Hkv == 0, f"{path}: {H} query heads not divisible by {Hkv} kv heads"
    if center_keys:
        k = k - k.mean(dim=1, keepdim=True)  # kbar: per-head mean over all positions
    if center_values:
        vmid = 0.5 * (v.max(dim=1, keepdim=True).values + v.min(dim=1, keepdim=True).values)
        v = v - vmid
    k_rep = k.repeat_interleave(H // Hkv, dim=0)  # same kv->q head map as repeat_kv

    q_norm = q.norm(dim=-1, p=2)
    Rq = q_norm.cummax(dim=-1).values
    Rk = k_rep.norm(dim=-1, p=2).cummax(dim=-1).values
    vmax = v.norm(dim=-1, p=torch.inf).cummax(dim=-1).values
    with torch.no_grad():
        log_Dnn = torch.stack([_causal_log_denominator(q[h], k_rep[h], scaling) for h in range(H)])
    return {
        "Rq": Rq.numpy(),
        "Rk": Rk.numpy(),
        "q_norm": q_norm.numpy(),
        "log_Dnn": log_Dnn.numpy(),
        "vmax": vmax.numpy(),
        "d": np.int64(q.shape[-1]),
        "scaling": np.float64(scaling),
        "center_keys": np.bool_(center_keys),
        "center_values": np.bool_(center_values),
    }


META_KEYS = ("d", "scaling", "center_keys", "center_values")


def load_derived(path, cache=True, scaling=None, center_keys=False, center_values=False):
    """Return the derived dict for one states file, computing and caching it if needed.

    scaling=None uses the layer's own temperature; a float forces that value
    (e.g. 1/sqrt(d)). Every (scaling, center_keys, center_values) combination
    gets its own cache file so switching between them does not recompute D_nn.
    """
    suffix = "" if scaling is None else f"__scaling={scaling:.6g}"
    suffix += "__ckeys" if center_keys else ""
    suffix += "__cvals" if center_values else ""
    cache_path = os.path.join(os.path.dirname(path), "derived", os.path.basename(path) + suffix + ".npz")
    if cache and os.path.exists(cache_path):
        with np.load(cache_path) as z:
            keys = set(z.files)
            if set(DERIVED_KEYS) <= keys and set(META_KEYS) <= keys and (
                scaling is None or np.isclose(float(z["scaling"]), scaling)
            ) and bool(z["center_keys"]) == center_keys and bool(z["center_values"]) == center_values:
                return {key: z[key] for key in (*DERIVED_KEYS, *META_KEYS)}
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is gone and no derived cache exists for scaling={scaling}, "
            f"center_keys={center_keys}, center_values={center_values} (looked for {cache_path})")
    out = _derive(path, scaling, center_keys, center_values)
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, **out)
    return out


def log_rho(z):
    """log rho_n = log n - log D_nn + s (||q_n||^2 + r_n^2) / 2, shape (H, n)."""
    s = float(z["scaling"])
    n = np.arange(1, z["log_Dnn"].shape[1] + 1, dtype=np.float64)
    return (np.log(n)[None, :] - z["log_Dnn"].astype(np.float64)
            + 0.5 * s * (z["q_norm"].astype(np.float64) ** 2 + z["Rk"].astype(np.float64) ** 2))


def load_rows(files, extract, tag="", cache=True, scaling=None, dtype=np.float32,
              center_keys=False, center_values=False):
    """Stack extract(derived) over all files into one (total_rows, N) array.

    Returns (arr, meta) where meta has the head dim "d", the temperature
    "scaling" and the "center_keys" / "center_values" flags. Prints progress
    every 20 files.
    """
    chunks = []
    meta = None
    for i, f in enumerate(files):
        z = load_derived(f, cache=cache, scaling=scaling, center_keys=center_keys, center_values=center_values)
        rows = np.asarray(extract(z), dtype=dtype)
        if meta is None:
            meta = {"d": int(z["d"]), "scaling": float(z["scaling"]),
                    "center_keys": bool(z["center_keys"]), "center_values": bool(z["center_values"])}
            N = rows.shape[1]
        assert int(z["d"]) == meta["d"], f"head dim mismatch in {f}: {int(z['d'])} != {meta['d']}"
        assert rows.shape[1] == N, f"n mismatch in {f}: {rows.shape[1]} != {N}"
        chunks.append(rows)
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"  {tag}: loaded {i + 1}/{len(files)} files", flush=True)
    return np.concatenate(chunks, axis=0), meta


def chunked_quantiles(arr, chunk=2048, qs=(0.25, 0.5, 0.75)):
    """np.quantile over axis 0, computed in column chunks to cap the sort buffer."""
    N = arr.shape[1]
    out = [np.empty(N, dtype=np.float64) for _ in qs]
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        res = np.quantile(arr[:, start:stop], list(qs), axis=0)
        for o, r in zip(out, res):
            o[start:stop] = r
    return tuple(out)
