"""Shared plotting utilities and parameters for longbench experiments."""

import matplotlib.pyplot as plt

#
# Plotting parameters
#
plt.rcParams.update(
    {
        # "font.family": "serif",
        # "font.serif": ["Liberation Serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

LABEL_FONT_SIZE = 7.75
XLABEL_FONT_SIZE = LABEL_FONT_SIZE
YLABEL_FONT_SIZE = LABEL_FONT_SIZE  # - 1.5
TICK_FONT_SIZE = 7
LEGEND_FONT_SIZE = 7
LINE_ALPHA = 0.75
LINE_WIDTH = 1
MARKER_ALPHA = 0.75
MARKER_SIZE = 3
OUTWARD = 4
NEURIPS_WIDTH = 5.5  # in inches
NEURIPS_HEIGHT = 2.25  # in inches

def get_method_style(method_name: str, h: float | None, causal: bool = True) -> dict[str, str]:
    """Return {"label", "color", "linestyle"} for a (method_name, h) pair.

    Lookup order: alias resolution → exact (method, h) match → wildcard
    (method, None) → fallback.
    """
    if causal:
        match method_name:
            case "hyper-cuda":
                return {
                    "label": "HyperAttention",
                    "color": "orange",
                    "linestyle": "-",
                    "marker": "o",
                    "hatch": "",
                }
            case "flash" | "exact":
                return {
                    "label": "Exact",
                    "color": "gray",
                    "linestyle": "--",
                    "marker": "o",
                    "hatch": "",
                }
            case "thinformer_triton":
                match h:
                    case 8:
                        return {
                            "label": "Thinformer Express ($h=8$)",
                            "color": "deepskyblue",
                            "linestyle": "-",
                            "marker": "s",
                            "hatch": "",
                        }
                    case 9:
                        return {
                            "label": "Thinformer Express ($h=9$)",
                            "color": "blue",
                            "linestyle": "-",
                            "marker": "D",
                            "hatch": "",
                        }
                    case 10:
                        return {
                            "label": "Thinformer Express ($h=10$)",
                            "color": "forestgreen",
                            "linestyle": "-",
                            "marker": "s",
                            "hatch": "",
                        }
                    case _:
                        raise ValueError(f"Unknown h for thinformer_triton: {h}")
            case _:
                raise ValueError(f"Not supported: method={method_name}, h={h}, causal={causal}")
    else:
        match method_name:
            case "thinformer_triton":
                return {
                    "label": "Thinformer (Triton)",
                    "color": "blue",
                    "linestyle": "-",
                    "marker": "s",
                    "hatch": "",
                }
            case "thinformer":
                return {
                    "label": "Thinformer (Original)",
                    "color": "red",
                    "linestyle": "-.",
                    "marker": "o",
                    "hatch": "",
                }
            case _:
                raise ValueError(f"Unknown method name: {method_name}")
        