"""Tests for thinformer_express (thinformer/triton/offline_express.py).

Settings cover the estimator's regimes:

  exact         -- S <= n_first + n_last + 4 * nout: early return, pure SDPA
  mixed         -- oversampled rounds (mstar == m) then undersampled rounds
  truncated     -- undersampling with a final round where (r - l) is not a
                   multiple of the subsampling step (floor'd strata count)
  mbar0         -- max_halving_rounds == 0: no compress rounds, warm coreset E
                   rides the off-band pass (n_warm)
  gqa           -- num_key_value_groups == 2 (query heads share KV heads)

Each setting checks shape/finiteness, bitwise determinism, and the exact-regime
prefix against SDPA. Undersampled rounds are additionally validated against a
dense per-query reference implementing the intended weighted estimator.
"""

import pytest
import torch

import thinformer.triton.offline_express as oe

E = 64
N_FIRST = N_LAST = 32
DELTA = 0.5
SEED = 0

# name -> (S, H, nout, mbar, num_key_value_groups)
SETTINGS = {
    "exact": (300, 2, 64, 4, 1),
    "mixed": (32768, 2, 64, 4, 1),
    "truncated": (32768, 2, 16, 2, 1),
    "mbar0": (4160, 2, 4, 0, 1),
    "gqa": (32768, 4, 64, 4, 2),
}


def make_inputs(
    S: int, H: int, kvg: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate seeded random q/k/v for a setting."""
    torch.manual_seed(SEED)
    q = torch.randn(1, S, H, E, device="cuda", dtype=torch.float16)
    k = torch.randn(1, S, H // kvg, E, device="cuda", dtype=torch.float16)
    v = torch.randn(1, S, H // kvg, E, device="cuda", dtype=torch.float16)
    return q, k, v


def run(
    setting: str, capture_undersampled: bool = False
) -> tuple[torch.Tensor, list[dict], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run thinformer_express for a named setting, optionally capturing
    the kwargs of every undersampled attention() call.
    """
    S, H, nout, mbar, kvg = SETTINGS[setting]
    q, k, v = make_inputs(S, H, kvg)
    captured = []
    original = oe.attention

    def capturing(**kw: object) -> None:
        res = original(**kw)
        if kw.get("log2_subsampling_factor", 0) > 0:
            torch.cuda.synchronize()
            captured.append(
                {
                    kk: (vv.clone() if torch.is_tensor(vv) else vv)
                    for kk, vv in kw.items()
                }
            )
        return res

    if capture_undersampled:
        oe.attention = capturing
    try:
        torch.manual_seed(SEED)
        out = oe.thinformer_express(
            query=q,
            key=k,
            value=v,
            nout=nout,
            max_halving_rounds=mbar,
            softmax_temp=E**-0.5,
            delta=DELTA,
            n_first=N_FIRST,
            n_last=N_LAST,
            num_key_value_groups=kvg,
            alias=f"test_{setting}",
        )
    finally:
        oe.attention = original
    torch.cuda.synchronize()
    return out, captured, (q, k, v)


@pytest.mark.parametrize("setting", SETTINGS)
def test_runs_finite_and_exact_prefix(setting: str) -> None:
    """Output has the right shape, is finite, and the exact-regime prefix
    (tokens through n_first + n_last + 4 * nout) matches plain SDPA.
    """
    S, H, nout, mbar, kvg = SETTINGS[setting]
    out, _, (q, k, v) = run(setting)
    assert out.shape == (1, S, H, E)
    assert torch.isfinite(out).all()

    exact_regime = min(S, N_FIRST + N_LAST + 4 * nout)
    ref = torch.nn.functional.scaled_dot_product_attention(
        query=q[:, :exact_regime].transpose(1, 2),
        key=k[:, :exact_regime].transpose(1, 2),
        value=v[:, :exact_regime].transpose(1, 2),
        scale=E**-0.5,
        is_causal=True,
        enable_gqa=(kvg > 1),
    ).transpose(1, 2)
    assert torch.equal(out[:, :exact_regime], ref)


@pytest.mark.parametrize("setting", SETTINGS)
def test_deterministic(setting: str) -> None:
    """Same seed, same code -> bitwise identical output."""
    out1, _, _ = run(setting)
    out2, _, _ = run(setting)
    assert torch.equal(out1, out2)


def test_mbar_none_rejected() -> None:
    """Uncapped depth (max_halving_rounds=None) is an unsupported
    configuration and must fail fast.
    """
    q, k, v = make_inputs(4160, 2, 1)
    with pytest.raises(AssertionError, match="max_halving_rounds"):
        oe.thinformer_express(
            query=q,
            key=k,
            value=v,
            nout=64,
            max_halving_rounds=None,
            softmax_temp=E**-0.5,
            delta=DELTA,
            n_first=N_FIRST,
            n_last=N_LAST,
            alias="test_none",
        )


def dense_reference(call: dict, i: int, b: int, h: int) -> torch.Tensor:
    """Intended weighted estimator for query rel-position i, batch b, query
    head h of a captured undersampled attention() call: softmax over
    [sink raw] + [window + on-band raw] + [off-band subsampled at weight
    2^log2_ss, incl. n_warm E points] + [level coresets at weight
    2^(log2_ss + 1 + level)].
    """
    kvg = call["num_key_value_groups"]
    hk = h // kvg
    qq = call["q"][b, h].float()
    kk = call["k"][b, hk].float()
    vv = call["v"][b, hk].float()
    level_idx = call["level_idx"][b, hk].long()
    iir = call["input_indices_round"][b, hk].long()
    bucket_size, l, r = call["bucket_size"], call["l"], call["r"]
    n_first, n_last = call["n_first"], call["n_last"]
    mstar, log2_ss, sm = call["m"], call["log2_subsampling_factor"], call["sm_scale"]
    n_warm = call.get("n_warm", 0)
    step = 1 << log2_ss
    sub_bucket = bucket_size >> log2_ss

    t_off = (i // sub_bucket) * sub_bucket
    ell = (i // bucket_size) * bucket_size
    n_off = n_warm + ((t_off - ell) >> log2_ss)
    lo, hi = n_first + l + t_off, n_first + n_last + l + i + 1

    gi = [torch.arange(0, n_first, device="cuda"), torch.arange(lo, hi, device="cuda")]
    gw = [torch.ones(n_first + hi - lo, device="cuda")]
    if n_off > 0:
        start = ell >> log2_ss
        gi.append(iir[start : start + n_off])
        gw.append(torch.full((n_off,), float(step), device="cuda"))
    t, cs, ll, ptr = i // bucket_size, bucket_size // step // 2, (r - l) // step, 0
    for level in range(mstar):
        n_lvl = (((t + 1) % 4) if level == mstar - 1 else (t % 4)) * cs
        if n_lvl > 0:
            o0 = ptr + (t // 4) * 4 * cs
            gi.append(level_idx[o0 : o0 + n_lvl])
            gw.append(
                torch.full((n_lvl,), float(2 ** (log2_ss + 1 + level)), device="cuda")
            )
        t //= 4
        cs *= 2
        level_bucket = sub_bucket * (2**level)
        ll = ll // level_bucket * level_bucket // 2
        ptr += ll
    gi, gw = torch.cat(gi), torch.cat(gw)
    logits = (qq[i] @ kk[gi].T) * sm + gw.log()
    return torch.softmax(logits, -1) @ vv[gi]


@pytest.mark.parametrize("setting", ["mixed", "truncated", "mbar0", "gqa"])
def test_undersampled_rounds_match_dense_reference(setting: str) -> None:
    """Kernel output on every captured undersampled round matches the dense
    per-query reference to fp16 accumulation noise.
    """
    _, captured, _ = run(setting, capture_undersampled=True)
    assert captured, f"setting {setting!r} produced no undersampled rounds"
    for c in captured:
        r_l = c["r"] - c["l"]
        torch.manual_seed(7)
        sample = torch.randint(0, r_l, (12,)).tolist() + [0, r_l - 1]
        B, H = c["q"].shape[0], c["q"].shape[1]
        for i in sample:
            for b in range(B):
                for h in range(H):
                    ref = dense_reference(c, i, b, h)
                    got = c["o"][b, h, i].float()
                    rel = (got - ref).norm() / ref.norm().clamp_min(1e-6)
                    assert rel < 5e-3, (
                        f"round l={c['l']} log2_ss={c['log2_subsampling_factor']} "
                        f"query i={i} head {h}: rel err {rel.item():.4f}"
                    )
