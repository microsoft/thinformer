"""Implementation of Express when the points are processed in a streaming manner."""

import math
import torch
import random
from dataclasses import dataclass, field

from .halve import halve_with_centering, halve_balancekv
from ..timing_utils import multi_events
from ..generic_utils import delta_compress, delta_warm

LOG2 = math.log(2)


@dataclass
class KV:
    """Data structure for storing key and value tensors.

    key (torch.Tensor): Key tensor of shape (B, S, H, E).

    value (torch.Tensor): Value tensor of shape (B, S, H, D).
    """

    key: torch.Tensor
    value: torch.Tensor

    def repeat(self, repeats: int) -> "KV":
        """Repeat the key and value tensors."""
        return KV(
            self.key.repeat(1, repeats, 1, 1), self.value.repeat(1, repeats, 1, 1)
        )

    # override slice operator
    def __getitem__(self, item: slice) -> "KV":
        """Slice the key and value tensors."""
        return KV(self.key[:, item], self.value[:, item])

    def __len__(self) -> int:
        """Compute the length of the key tensor."""
        return self.key.shape[1]


#
# Streaming KH-Compress
#


def print_coreset(coreset: list) -> None:
    """Print the coreset.

    Args:
        coreset (list): List of tuples (KV, level) where KV is a KV object
            and level is the compression level.

    """
    print("Coreset:")
    for kv, level in coreset:
        print(f"  Level {level}: {len(kv)} points")
        print(f"    Key shape: {kv.key.shape}, Value shape: {kv.value.shape}")


@dataclass
class Level:
    """Data structure for storing the key and value tensors at each compression level."""

    data: list[KV] = field(default_factory=list)

    def __len__(self) -> int:
        """Compute the length of the concatenated tensors."""
        return sum(x.key.shape[1] for x in self.data)

    def cat_clear(self) -> KV:
        """Concatenate the elements in the level and clear."""
        assert len(self.data) > 0, "Cannot concatenate empty level"
        result = self.cat()
        self.data.clear()
        return result

    def cat(self) -> KV:
        """Concatenate the elements in the level without clearing."""
        assert len(self.data) > 0, "Cannot concatenate empty level"
        keys = [x.key for x in self.data]
        values = [x.value for x in self.data]
        return KV(torch.cat(keys, dim=1), torch.cat(values, dim=1))

    def append(self, kv: KV) -> None:
        """Append a KV pair to the level."""
        self.data.append(kv)


class StreamingCompress(object):
    """Data structure for storing the Streaming Compress levels."""

    def __init__(
        self,
        nin: int,
        m: int,
        halve_probs: list[float],
        alias: str = "khcompress",
        softmax_temp: float = 1.0,
        M_inner: int = 64,
        use_balancekv: bool = False,
    ):
        """Args:
        nin (int): number of input points we expect.
        m (int): number of halving rounds.
        halve_probs (list[float]): halving probability for each level 0..m-1.
        alias (str): Alias for timings.
        softmax_temp (float): Softmax temperature for the exp kernel.
        use_balancekv (bool): If True, halve with BalanceKV's balanced walk
            instead of Kernel Halving.

        """
        assert len(halve_probs) == m, (
            f"Expected {m} halve probabilities, got {len(halve_probs)}."
        )
        # Initialize parameters
        self.m = m
        self.nin = nin
        # Halve parameters
        self.halve_probs = halve_probs
        self.softmax_temp = softmax_temp
        self.M_inner = M_inner
        self.use_balancekv = use_balancekv
        # Logging parameters
        self.alias = alias

        # Initialize streaming data structure
        self.levels = [Level() for _ in range(m + 1)]
        self.num_halve_calls = [0 for _ in range(m)]

    def __call__(
        self,
        kv: KV,
        b_sqd: torch.Tensor = None,
        uniforms: dict[int, torch.Tensor] = None,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the compression levels with the given key and value tensors.

        Args:
            kv (KV): Key and value tensors to compress.
                Must have size 4^(g+1) in the second dimension.
            shift (torch.Tensor): Shift parameter for the exp kernel.
                If None, it will be computed from the input key tensor.
            b_sqd (torch.Tensor): b_sqd parameter for the exp kernel.
                If None, it will be computed from the input value tensor.
            uniforms (dict[int, torch.Tensor]): Uniform random variables for each halving round.
            verbose (bool): If True, print verbose logs.
            halve_prob (float): if provided, it overrides the default halving probability.
            vbar (torch.Tensor): vbar parameter for the exp kernel.
            kbar (torch.Tensor): kbar parameter for the exp kernel.

        Returns:
            list of tuples: Each tuple contains a KV pair and the level index.
                The level index equals log2 the number of times the key and value tensors
                need to be repeated to reconstruct the final coreset.

        """
        batch_size = self.nin * 4 // 4**self.m
        # Add kv to the 0-th level
        self.levels[0].append(kv)
        # Iterate over levels and compress
        for l in range(self.m):
            if len(self.levels[l]) >= batch_size * 2**l:
                # If the first level is full, we need to compress it
                kv_0 = self.levels[l].cat_clear()
                # Unpack key and value tensors
                key = kv_0.key
                value = kv_0.value
                if self.use_balancekv:
                    compressed_key, compressed_value = halve_balancekv(
                        key,
                        value,
                    )
                else:
                    compressed_key, compressed_value = halve_with_centering(
                        key,
                        value,
                        halve_prob=self.halve_probs[l],
                        softmax_temp=self.softmax_temp,
                        alias=f"{self.alias}__halve-l{l}h{len(self.levels[l])}",
                        b_sqd=b_sqd,
                        uniforms=uniforms[f"{self.alias}__halve-l{l}"][
                            :, self.num_halve_calls[l] : self.num_halve_calls[l] + 1
                        ]
                        if uniforms is not None
                        else None,
                    )
                self.levels[l + 1].append(KV(compressed_key, compressed_value))
                self.num_halve_calls[l] += 1

        coreset = []
        # Collect intermediate coresets from each level to construct the output coreset.
        # NOTE: We iterate from the last level to the first level
        # to ensure that the most recent points are at the end of the coreset
        for l in range(self.m, -1, -1):
            if len(self.levels[l]) > 0:
                # l = log2 number of repeats
                coreset.extend([(kv, l) for kv in self.levels[l].data])
        return coreset

    def flush_top(self) -> KV:
        """Return the completed top-level coreset (size nin / 2^m) and clear it."""
        assert all(len(self.levels[l]) == 0 for l in range(self.m)), (
            f"Lower levels must be empty at sibling completion, got sizes "
            f"{[len(self.levels[l]) for l in range(self.m + 1)]}."
        )
        return self.levels[self.m].cat_clear()


class Express(object):
    """Streaming Express (algo:express in express.tex).

    Each round m (even) processes three sibling groups of 2^m * nout points.
    Each sibling is thinned to nout points via Subsample(m, mbar) +
    Compress2(m ^ mbar) and appended to the Express coreset E. Once E reaches
    4 * nout points, it is halved twice back down to nout and m advances by 2.

    NOTE (round m=0 vs the paper): algo:express processes the first nout
    points in the exact phase and then feeds each of the three nout-sized
    round-0 siblings through Subsample(0, mbar) + Compress2(0) before
    appending it to E. Both of those are identity maps at m=0 (Subsample
    keeps everything when m <= mbar, and Compress2(0) has no halving levels),
    so this implementation treats the whole of round 0 as a no-op instead:
    all 4 * nout raw points are ingested in a single call and appended to E
    as raw nout-blocks, without instantiating Subsample/Compress2. The cache
    state after any prefix of the stream is identical to the paper's; only
    the bookkeeping is batched.
    """

    def __init__(
        self,
        nout: int,
        delta: float = 0.5,
        softmax_temp: float = 1.0,
        alias: str = "express",
        max_halving_rounds: int = None,
        use_balancekv: bool = False,
    ):
        """Args:
        nout (int): Nominal coreset size.
        delta (float): Base failure probability.
        softmax_temp (float): Softmax temperature.
        alias (str): Alias for timings.
        max_halving_rounds (int): Cap mbar on the Compress2 halving depth.
            If None, use the largest m such that 2^m divides nout.
        use_balancekv (bool): If True, every halving step (both the halve
            phase and Compress2) uses BalanceKV's balanced walk instead of
            Kernel Halving.

        """
        self.nout = nout
        self.delta = delta
        self.alias = alias
        self.softmax_temp = softmax_temp
        self.use_balancekv = use_balancekv

        # number of tokens seen so far
        self.n = 0
        # Express coreset E: list of (KV, level) chunks, all at level self.m
        self.E = []
        # Weighted cache of the active sibling's Compress2 instance
        self.compress_cache = []

        # counter to keep track of the previous cache size
        self.size = 0

        # subsampling implementation
        if max_halving_rounds is None:
            # find the largest m such that nout is divisible by 2^m
            self.max_halving_rounds = (nout & -nout).bit_length() - 1
        else:
            self.max_halving_rounds = max_halving_rounds
        assert nout % 2 ** max(self.max_halving_rounds - 1, 0) == 0, (
            f"nout={nout} must be divisible by "
            f"2^(mbar-1)={2 ** max(self.max_halving_rounds - 1, 0)}."
        )

        # Round state: m is the (even) thinning factor, ell counts the points
        # consumed by the active sibling, n_to_keep is the reservoir index of
        # Subsample in the capped regime (m > mbar).
        self.m = 0
        self.ell = 0
        self.n_to_keep = None
        self.compress = None

    def __call__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        b_sqd: torch.Tensor = None,
        alias: str = "express",
        uniforms: dict = None,
        verbose: bool = False,
    ):
        """Invoke Express on a bucket of key and value tensors.

        Args:
            key (torch.Tensor): Key tensor of shape (B, bucket_size, H, D).
            value (torch.Tensor): Value tensor of shape (B, bucket_size, H, D).
            b_sqd (torch.Tensor): Squared norm of the value tensor.
                If None, it will be computed from the value tensor.
            verbose (bool): If True, print verbose information.
            alias (str): Alias for the event.
            uniforms (dict): Dict of uniform RV buffers for each halving round.

        Yields:
            compressed_key (torch.Tensor): Compressed key tensor.
                NOTE: this is an unweighted coreset with repeats.
            compressed_value (torch.Tensor): Compressed value tensor.
                NOTE: this is an unweighted coreset with repeats.
            bucket_size (int): Number of points consumed in the current batch.

        """
        B, bucket_size, H, D = key.shape
        # import pdb; pdb.set_trace()
        if self.n + bucket_size <= self.nout:
            # In the first nout tokens, we simply return the input as the coreset
            self.E.append((KV(key, value), 0))
            self.n += bucket_size
            self.size = self.n
            return key, value, bucket_size, None

        # --- Prologue: sibling boundary ---
        if self.ell == 0:
            # Halve phase: E holds the previous round's coreset plus its three
            # sibling outputs (4 chunks of nout points, all at level m); halve
            # it twice down to nout and advance m by 2 (algo:express L63-64).
            E_size = sum(len(x) for x, _ in self.E)
            if E_size == 4 * self.nout:
                cat_key = torch.cat([x.key for x, _ in self.E], dim=1)
                cat_value = torch.cat([x.value for x, _ in self.E], dim=1)
                # Both halve-phase calls are halve_{m,m} with pre-increment m
                halve_prob = delta_warm(self.delta, self.m)
                for suffix in ["a", "b"]:
                    if self.use_balancekv:
                        cat_key, cat_value = halve_balancekv(
                            cat_key,
                            cat_value,
                            verbose=verbose,
                        )
                    else:
                        cat_key, cat_value = halve_with_centering(
                            cat_key,
                            cat_value,
                            halve_prob=halve_prob,
                            softmax_temp=self.softmax_temp,
                            alias=f"{alias}__halve-m{self.m}{suffix}",
                            b_sqd=b_sqd,
                            uniforms=uniforms[f"{alias}__halve-m{self.m}{suffix}"]
                            if uniforms is not None
                            else None,
                        )
                self.m += 2
                cset = KV(cat_key, cat_value)
                assert len(cset) == self.nout, (
                    f"E must have length {self.nout} after the halve phase, "
                    f"but got {len(cset)}."
                )
                self.E = [(cset, self.m)]
            # Reset thinning objects: Subsample(m, mbar) + Compress2(m ^ mbar)
            mstar = min(self.m, self.max_halving_rounds)
            self.compress = StreamingCompress(
                nin=self.nout * 2**mstar,
                m=mstar,
                halve_probs=[
                    delta_compress(self.delta, self.m, i, mstar) for i in range(mstar)
                ],
                alias=f"{alias}__compress-m{self.m}",
                softmax_temp=self.softmax_temp,
                use_balancekv=self.use_balancekv,
            )
            self.compress_cache = []
            self.n_to_keep = None

        # --- Main body ---
        kv = KV(key, value)
        # In the capped regime (m > mbar), each point fed to Compress2 stands in
        # for a group of G = 2^(m - mbar) raw points, shifting all levels up.
        shift = max(self.m - self.max_halving_rounds, 0)
        # The attention coreset reflects the cache before ingesting kv
        coreset = self.E + [(x, l + shift) for x, l in self.compress_cache] + [(kv, 0)]
        if self.m == 0:
            # Round 0: siblings are raw nout-blocks, so append them to E
            # directly. The paper feeds each sibling through Subsample(0, mbar)
            # + Compress2(0), but both are identity maps at m=0 (see the class
            # docstring), so we skip them entirely. kv may span several
            # siblings (the whole round arrives in one 3*nout call) and need
            # not be chunk-aligned; only the total E size matters for the
            # halve phase.
            for lo in range(0, len(kv), self.nout):
                self.E.append((kv[lo : lo + self.nout], 0))
        elif shift == 0:
            self.compress_cache = self.compress(
                kv, b_sqd=b_sqd, uniforms=uniforms, verbose=verbose
            )
            self.ell += len(kv)
        else:
            # Subsample: keep one uniform index per group of G points
            assert len(kv) == 1, f"Capped regime expects bucket size 1, got {len(kv)}."
            G = 2**shift
            if self.ell % G == 0:
                self.n_to_keep = self.ell + random.randrange(G)
            if self.ell == self.n_to_keep:
                self.compress_cache = self.compress(
                    kv, b_sqd=b_sqd, uniforms=uniforms, verbose=verbose
                )
            self.ell += 1

        # Sibling completion: append the finished Compress2 output to E
        if self.ell == self.nout * 2**self.m:
            cset = self.compress.flush_top()
            assert len(cset) == self.nout, (
                f"Sibling output must have length {self.nout}, but got {len(cset)}."
            )
            self.E.append((cset, self.m))
            self.ell = 0
            self.compress_cache = []

        # Construct compressed key and value tensors
        assert all(l1 >= l2 for (_, l1), (_, l2) in zip(coreset, coreset[1:])), (
            "Levels in the coreset must be in descending order."
        )
        # Convert the coreset list into contiguous key and value tensors
        with multi_events(f"{alias}_cat"):
            compressed_key = torch.cat([x.key for x, l in coreset], dim=1)
            compressed_value = torch.cat([x.value for x, l in coreset], dim=1)
        with multi_events(f"{alias}_log-weights"):
            log_key_weights = torch.zeros(
                (B, compressed_key.shape[1], H),
                device=key.device,
                dtype=key.dtype,
            )
            coreset_size = 0
            for x, l in coreset:
                log_key_weights[:, coreset_size : coreset_size + x.key.shape[1], :] = (
                    l * LOG2
                )
                coreset_size += x.key.shape[1]
        self.n += len(kv)  # update actual number of points seen so far

        # update the coreset size
        self.size = compressed_key.shape[1]
        return compressed_key, compressed_value, len(kv), log_key_weights

    def get_bucket_size(self) -> int:
        """Get the desired size of the current level-0 coreset assuming we have processed self.n points.

        Returns:
            bucket_size (int): number of points in the current bucket.

        """
        if self.n < 4 * self.nout:
            # Round m=0 (including the initial nout exact points): the paper
            # processes one nout (exact phase) then three nout siblings, but
            # Subsample(0, mbar) and Compress2(0) are identity maps, so the
            # whole round [0, 4*nout) is consumed as one no-op call (see the
            # Express class docstring). This matches express_2.py's round-0
            # bucket of 4*nout raw points so the two driver loops run the
            # same number of iterations.
            return 4 * self.nout
        # m = 2 * floor(log4(n / nout)), the (even) thinning factor of the round
        # that the next point belongs to
        i = (self.n // self.nout).bit_length() - 1
        m = i & ~1
        if m > self.max_halving_rounds:
            # Capped regime: reservoir Subsample needs per-point decisions
            return 1
        # Level-0 batch size of Compress2(m): 2^m * nout / 4^(m-1)
        return (4 * self.nout) >> m

    def invoke(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        b_sqd: torch.Tensor = None,
        verbose: bool = False,
        alias: str = "express",
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Invoke Express on any number of KV points (not necessarily a bucket size).

        Args:
            key (torch.Tensor): Key tensor of shape (B, S, H, D).
            value (torch.Tensor): Value tensor of shape (B, S, H, D).
            b_sqd (torch.Tensor): Squared norm of the value tensor.
                If None, it will be computed from the value tensor.
            verbose (bool): If True, print verbose information.
            alias (str): Alias for the event.

        Returns:
            compressed_key (torch.Tensor): Compressed key tensor.
            compressed_value (torch.Tensor): Compressed value tensor.
            bucket_size (int): Number of points consumed in the current batch.
            log_key_weights (torch.Tensor): Log of key weights.

        """
        B, S, H, D = key.shape
        bucket_size = self.get_bucket_size()
        # set r to be the number of points needed to complete a bucket
        l = 0
        r = bucket_size - (self.n % bucket_size)
        while l < S:
            if verbose:
                print(f"l: {l}, r: {r}")
                # import pdb; pdb.set_trace()
            compressed_key, compressed_value, _, _log_key_weights = self(
                key[:, l:r],
                value[:, l:r],
                b_sqd=b_sqd,
                verbose=verbose,
                alias=alias,
            )
            # advance the left and right pointers to the next bucket boundary
            # (recompute the remainder each iteration: bucket sizes change at
            # round boundaries, e.g. nout -> 4*nout entering round m=0)
            l = r
            bucket_size = self.get_bucket_size()
            r = l + bucket_size - (self.n % bucket_size)
        return compressed_key, compressed_value, None, _log_key_weights
