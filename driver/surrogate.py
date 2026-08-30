# SPDX-License-Identifier: Apache-2.0
"""Gradient-enhanced surrogate fitted to optimizer exhaust: (J, grad J) pairs per objective.

Opportunistic accelerator for globalizing the sweep, cheap UQ, and warm starts.
Not on the critical path.
"""


class GradientEnhancedSurrogate:
    def fit(self, x, values, grads):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
