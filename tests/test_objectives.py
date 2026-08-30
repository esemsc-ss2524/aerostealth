# SPDX-License-Identifier: Apache-2.0
import numpy as np

from analysis.pareto import nondominated_mask, sorted_front
from driver.objectives import ks_aggregate, normalize


def test_ks_brackets_max_and_tightens_with_rho():
    v = np.array([0.1, 0.4, 0.2, 0.05])
    loose = float(ks_aggregate(v, 5.0))
    tight = float(ks_aggregate(v, 200.0))
    assert v.max() <= loose
    assert abs(tight - v.max()) < abs(loose - v.max())


def test_normalize():
    assert normalize(8.0e-3, 8.0e-3) == 1.0


def test_nondominated_mask():
    pts = np.array([[1.0, 4.0], [2.0, 2.0], [3.0, 1.0], [2.5, 3.0], [4.0, 5.0]])
    assert nondominated_mask(pts).tolist() == [True, True, True, False, False]


def test_sorted_front_is_ordered_by_first_objective():
    pts = np.array([[3.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
    front = sorted_front(pts)
    assert np.all(np.diff(front[:, 0]) > 0)
