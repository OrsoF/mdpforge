import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge.core.operators import compute_transition_reward_policy
from mdpforge.core.partition import Partition
from mdpforge.utils.projected_bellman import (
    apply_pobo_until_var_small,
    apply_poqbo_until_var_small,
    apply_ppbo_until_var_small,
)


@pytest.mark.parametrize("discount", [0.9, 1.0])
def test_projected_iterations_recover_chain_values(chain, discount):
    partition = Partition(chain, labels=np.arange(chain.state_dim))
    transitions, rewards = partition.compute_agg_trans_reward_v()
    value, _ = apply_pobo_until_var_small(
        chain,
        discount,
        transitions,
        rewards,
        partition.weights,
        1e-9,
        max_steps=1000,
        shift_acceleration=True,
    )
    expected = [2 * discount, 2, 0]
    assert_allclose(value, expected, atol=1e-8, rtol=0)

    transitions, rewards = partition.compute_agg_trans_reward_q()
    q_value, _ = apply_poqbo_until_var_small(
        chain,
        discount,
        transitions,
        rewards,
        1e-9,
        max_steps=1000,
        shift_acceleration=True,
    )
    assert q_value.shape == chain.reward_matrix.shape
    assert_allclose(q_value.max(axis=1), expected, atol=1e-8, rtol=0)

    transition, reward = compute_transition_reward_policy(chain, np.zeros(3, dtype=int))
    transition, reward = partition.compute_agg_trans_reward_pi(transition, reward)
    policy_value, _ = apply_ppbo_until_var_small(
        discount,
        transition,
        reward,
        1e-9,
        np.zeros(3),
        1000,
        shift_acceleration=True,
    )
    assert_allclose(policy_value, expected, atol=1e-8, rtol=0)
