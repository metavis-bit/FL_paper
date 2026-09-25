"""Parameter-free privacy cost used by Adap-CTA decision making."""

import math


def privacy_cost(correction_ratio: float) -> float:
    """Return ``exp(-ratio)`` for a correction ratio in ``[0, 1]``."""

    ratio = min(max(float(correction_ratio), 0.0), 1.0)
    return math.exp(-ratio)
