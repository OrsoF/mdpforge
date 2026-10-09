from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdpforge.core.partition import Partition
from mdpforge.core.validation import validate_model


def test_partition_regions_cover_every_state(small_model):
    partition = Partition(small_model, [10, 10, 3, 8])
    regions = partition.states_in_region
    assert all(regions)
    assert_array_equal(
        np.sort(np.concatenate(regions)), np.arange(small_model.state_dim)
    )
    assert_array_equal(np.unique(partition.state_to_region), np.arange(3))
    for label, states in enumerate(regions):
        assert np.all(partition.state_to_region[states] == label)


def test_partition_weights_sum_correctly(small_model):
    partition = Partition.from_regions(small_model, [[0, 1], [2], [3]])
    assert_allclose(
        partition.weights.toarray(),
        [[0.5, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    )
    assert_allclose(np.asarray(partition.weights.sum(axis=1)).ravel(), np.ones(3))
    assert_allclose(np.asarray(partition.phi.sum(axis=1)).ravel(), np.ones(4))
    assert_allclose((partition.weights @ partition.phi).toarray(), np.eye(3))
    assert_allclose(partition.phi @ np.array([2, 5, 7]), [2, 2, 5, 7])
    assert_allclose(partition.weights @ np.array([0, 2, 4, 6]), [1, 4, 6])


def test_aggregated_model_remains_stochastic(small_model):
    partition = Partition(small_model, [0, 0, 1, 1])
    transitions, rewards = partition.compute_agg_trans_reward_q()
    validate_model(
        SimpleNamespace(
            name=f"aggregated_{small_model.name}",
            state_dim=2,
            action_dim=small_model.action_dim,
            transition_matrix=transitions,
            reward_matrix=rewards,
        )
    )
    assert_allclose(rewards, [[1, 0.75], [2.5, 1.75]])


@pytest.mark.parametrize("refinement", ["width", "tiles"])
def test_refinement_never_merges_existing_regions(refinement):
    model = SimpleNamespace(state_dim=8)
    partition = Partition(model, [0, 0, 0, 0, 1, 1, 1, 1])
    original_labels = partition.state_to_region.copy()
    value = np.array([0, 0.1, 0.6, 0.7, 0, 0.1, 0.6, 0.7])
    assert [len(region) for region in partition.states_in_region] == [4, 4]
    assert_allclose(partition.phi @ np.array([5, 9]), [5, 5, 5, 5, 9, 9, 9, 9])
    assert_allclose(partition.weights @ value, [0.35, 0.35])
    if refinement == "width":
        partition.refine_by_width(value, 0.5)
        assert np.all(partition.span(value) < 0.5)
    else:
        partition.refine_by_tiles(value, 2)
    regions = partition.states_in_region
    assert partition.n_regions == 4
    assert_array_equal(np.sort(np.concatenate(regions)), np.arange(model.state_dim))
    for region in regions:
        assert np.unique(original_labels[region]).size == 1
    # Projection must reflect the refined regions even after prior use.
    assert_allclose((partition.weights @ partition.phi).toarray(), np.eye(4))
