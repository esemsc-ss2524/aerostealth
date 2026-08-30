# SPDX-License-Identifier: Apache-2.0
"""Multiple-gradient descent to trace the Cd vs sigma_agg front continuously."""


def common_descent_direction(grads):
    """Min-norm point in the convex hull of the objective gradients (Frank-Wolfe)."""
    raise NotImplementedError


def trace_front(forward, x0, steps):
    raise NotImplementedError
