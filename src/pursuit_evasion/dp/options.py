"""Reject invalid integration options before allocation or compiler invocation."""

import math


def validate_options(horizon, output_dt, order, cfl, threads):
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Horizon must be finite and positive.")
    if not math.isfinite(output_dt) or output_dt <= 0:
        raise ValueError("Output spacing must be finite and positive.")
    if not math.isfinite(cfl) or not 0 < cfl <= 1:
        raise ValueError("CFL must be finite and in (0, 1].")
    if order not in (1, 2):
        raise ValueError("Supported numerical orders are 1 and 2.")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("Thread count must be a positive integer.")
