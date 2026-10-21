"""Uniform-shift certification is a property of every discounted MDP."""

import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdpforge.core.operators import bellman_operator
from mdpforge.core.precision import certify_value
from tests._contracts import enumerate_optimal_value


@pytest.mark.parametrize("discount", [0.5, 0.9])
@pytest.mark.parametrize("shift", [-37.0, 37.0])
def test_certification_removes_uniform_shift_and_preserves_policy(
    mdp_case, discount, shift
):
    expected = enumerate_optimal_value(mdp_case, discount)
    candidate = expected + shift
    policy = bellman_operator(mdp_case, candidate, discount).argmax(axis=1)
    certified = certify_value(mdp_case, candidate, discount, 1e-6)
    assert_allclose(certified, expected, atol=1e-6, rtol=0)
    assert_array_equal(
        bellman_operator(mdp_case, certified, discount).argmax(axis=1), policy
    )
    assert_allclose(candidate, expected + shift, atol=0, rtol=0)


@pytest.mark.parametrize("discount", [0.5, 0.9])
def test_certification_rejects_values_with_nonconstant_bellman_error(
    mdp_case, discount
):
    inaccurate = enumerate_optimal_value(mdp_case, discount).copy()
    inaccurate[0] += 1
    with pytest.raises(RuntimeError, match="VI precision not reached"):
        certify_value(mdp_case, inaccurate, discount, 1e-4)
