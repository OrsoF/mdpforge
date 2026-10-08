import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge.core.operators import bellman_operator
from mdpforge.core.precision import certify_value


def test_certification_removes_uniform_offset_without_changing_policy(chain):
    value = np.array([1.8, 2, 0]) + 37
    policy = bellman_operator(chain, value, 0.9).argmax(axis=1)
    certified = certify_value(chain, value, 0.9, 1e-6)
    assert_allclose(certified, [1.8, 2, 0], atol=1e-6, rtol=0)
    assert np.array_equal(
        bellman_operator(chain, certified, 0.9).argmax(axis=1), policy
    )


def test_certification_rejects_inaccurate_values(chain):
    with pytest.raises(RuntimeError, match="VI precision not reached"):
        certify_value(chain, np.zeros(3), 0.9, 1e-3)
