"""General utility functions for Thinformer."""

import math


def delta_m(delta: float, m: int) -> float:
    """Base failure probability at Express level m (Proposition 1, khexpress-guarantees).

    delta_m = (delta / 2) * (1 / log_2(m/2 + 2) - 1 / log_2(m/2 + 3))

    The sum over even m >= 0 telescopes to (delta / 2) * 1/log_2(2) = delta / 2.
    """
    assert m >= 0 and m % 2 == 0, f"m must be a non-negative even integer, got {m}"
    return 0.5 * delta * (1.0 / math.log2(m / 2 + 2) - 1.0 / math.log2(m / 2 + 3))


def delta_compress(delta: float, m: int, i: int, mstar: int) -> float:
    """Compress2 halving (m, i) failure probability for i in [0, m-1], m >= 2 even."""
    assert m >= 2 and m % 2 == 0 and 0 <= i < m, (
        f"need m even >= 2 and 0 <= i < m, got m={m}, i={i}"
    )
    return (4 ** (i + 1 - mstar)) * delta_m(delta, m) / (3 * mstar)


def delta_warm(delta: float, m: int) -> float:
    """Warm-start halve probability at level m (each of inner & outer halves)."""
    return delta_m(delta, m) / 2


def log4_largest_power_of_four(n: int) -> int:
    """Returns log_4 of the largest power of four less than or equal to n.

    Args:
        n: An integer input.

    Returns:
        Log base 4 of the largest power of four less than or equal to n.

    """
    return (n.bit_length() - 1) // 2


def decimal_to_base4(decimal_num: int) -> list[int]:
    """Convert a decimal number to a list of base 4 digits.

    Args:
        decimal_num: A decimal number.

    Returns:
        A list of base 4 digits.

    """
    if decimal_num == 0:
        return [
            0,
        ]
    base4_digits = []
    while decimal_num > 0:
        base4_digits.insert(0, decimal_num % 4)
        decimal_num //= 4
    return base4_digits


def get_coreset_size(
    S: int, g: int, halve_remainder: bool = False, inflation_factor: int = 1
) -> int:
    """Get the coreset size for KH-Compress.

    Args:
        S: The input sequence length.
        g: The oversampling factor.
        halve_remainder: If True, halve the remainder of size S % 4^{g+1}.
        inflation_factor: The inflation factor for the initial bucket size.

    Returns:
        The coreset size.

    """
    # import pdb; pdb.set_trace()
    assert inflation_factor in [1, 2], "inflation_factor must be 1 or 2"
    batch_size = 4 ** (g + 1) * inflation_factor
    # Consider the base 4 representation of S
    base4 = decimal_to_base4(S // batch_size)
    remainder = S % batch_size
    # the last g+1 digits are not compressed, which we call the remainder
    coreset_size = 0
    for i, d in enumerate(base4[::-1]):
        coreset_size += d * (4**g * inflation_factor) * 2 ** (i + 1)
    if halve_remainder:
        coreset_size += (remainder + 1) // 2
    else:
        coreset_size += remainder
    return coreset_size
