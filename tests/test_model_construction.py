"""Small scalar oracles for optimised catalogue model construction."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import binom

from mdpforge.core.validation import validate_model
from mdpforge.models import dam, inventory, tandem, tandem_choice
from mdpforge.utils import persistence


@pytest.mark.parametrize("module", [tandem, tandem_choice])
@pytest.mark.parametrize("state_dim", [16, 81])
@pytest.mark.parametrize("service_rates", [(0.2, 0.2), (0.0, 0.3)])
def test_tandem_construction_matches_scalar_events_and_costs(
    module, state_dim, service_rates
):
    model = module.Model(state_dim=state_dim)
    model.mu_1, model.mu_2 = service_rates
    model.lambda_tilde = (
        model.arrival_rate_lambda + model.K1 * model.mu_1 + model.K2 * model.mu_2
    )
    model.create_model(cache="off")
    validate_model(model)
    for action in range(model.action_dim):
        expected = np.zeros((model.state_dim, model.state_dim))
        for state, (m1, m2, k1, k2) in enumerate(model.state_list):
            if module is tandem:
                a1, a2 = model.action_decoding[action]
                chosen_1 = model.n1(k1 + a1)
                chosen_2 = model.n2(k2 + a2)
            else:
                chosen_1, chosen_2 = model.action_list[action]
            rates = (
                model.arrival_rate_lambda,
                model.mu_1 * min(m1, chosen_1),
                model.mu_2 * min(m2, chosen_2),
            )
            pseudo = model.lambda_tilde - (rates[0] + rates[1] + rates[2])
            if module is tandem_choice:
                pseudo = max(0.0, pseudo)
            targets = (
                model.s1p(state, action),
                model.s2p(state, action),
                model.s3p(state, action),
                state,
            )
            for target, rate in zip(targets, (*rates, pseudo)):
                expected[state, target] += (
                    rate / model.lambda_tilde if module is tandem_choice else rate
                )
            assert model.reward_matrix[state, action] == -model.Reward_tilde(
                state, action
            )
        if module is tandem:
            expected /= model.lambda_tilde
        assert_array_equal(model.transition_matrix[action].toarray(), expected)


@pytest.mark.parametrize("requested", [4, 12])
@pytest.mark.parametrize(
    "demand_size, probability", [(0, 0.4), (2, 0.0), (2, 1.0), (5, 0.4)]
)
def test_inventory_preserves_saturated_demand_and_rewards(
    requested, demand_size, probability
):
    model = inventory.Model(state_dim=requested, action_dim=4)
    model.BinomialeQ, model.BinomialeP = demand_size, probability
    model.create_model(cache="off")
    validate_model(model)
    assert model.requested_state_dim == requested
    assert model.name == f"{model.state_dim}_{model.action_dim}_inventory"
    for action in range(model.action_dim):
        expected = np.zeros((model.state_dim, model.state_dim))
        for state in range(model.state_dim):
            after_order = min(state + action, model.state_dim - 1)
            for demand in range(demand_size + 1):
                expected[state, max(0, after_order - demand)] += binom.pmf(
                    demand, demand_size, probability
                )
            stock = state - model._stock_size
            cost = (
                -model._shortage_cost * stock
                if stock < 0
                else model._holding_cost * stock
            )
            assert model.reward_matrix[state, action] == (
                cost + action * model._action_cost
            )
        assert_array_equal(model.transition_matrix[action].toarray(), expected)


def test_inventory_can_load_its_complete_state_from_cache(
    isolated_model_cache, monkeypatch
):
    built = inventory.Model(state_dim=12, action_dim=4)
    built.create_model(cache="readwrite")
    assert persistence.model_cache_path(built).is_file()

    def forbidden_build(self):
        pytest.fail("A matching deterministic inventory must use its cache")

    monkeypatch.setattr(inventory.Model, "_build_model", forbidden_build)
    restored = inventory.Model(state_dim=12, action_dim=4)
    restored.create_model(cache="readonly")
    assert restored.get_config() == built.get_config()
    assert_array_equal(restored.reward_matrix, built.reward_matrix)
    for actual, expected in zip(restored.transition_matrix, built.transition_matrix):
        assert_array_equal(actual.toarray(), expected.toarray())


@pytest.mark.parametrize("state_dim", [36, 64])
def test_dam_preserves_scalar_flows_and_seeded_draw_order(state_dim):
    reference_rng = np.random.default_rng(921)
    model = dam.Model(state_dim=state_dim, seed=921)
    model.create_model(cache="off")
    validate_model(model)
    capacity = model._max_dam_capa
    for action_index, action in enumerate(model.action_list):
        expected = np.zeros((model.state_dim, model.state_dim))
        for state_index, state in enumerate(model.state_list):
            for flow in range(capacity + 1):
                volumes, outgoing = [], []
                for index, release in enumerate(action):
                    inflow = flow + (state[model._n_dam + index - 1] if index else 0)
                    volume, turbined, spilled = dam.update_first_dam(
                        state[index], inflow, release, capacity, capacity, capacity
                    )
                    volumes.append(volume)
                    outgoing.append(min(turbined + spilled, capacity))
                target = model.state_dict[tuple(volumes + outgoing)]
                expected[state_index, target] += 1.0
        expected /= capacity + 1
        assert_array_equal(model.transition_matrix[action_index].toarray(), expected)
        assert_array_equal(
            model.reward_matrix[:, action_index],
            sum(action) * reference_rng.random(model.state_dim),
        )
    assert_array_equal(
        model.transition_matrix_normalization,
        np.full((model.state_dim, model.action_dim), float(capacity + 1)),
    )


@pytest.mark.parametrize("seed", [0, np.int64(23)])
def test_dam_seed_is_reproducible_and_independent_of_build_order(seed):
    global_state = np.random.get_state()
    first = dam.Model(state_dim=64, seed=seed)
    first.create_model(cache="off")
    other = dam.Model(state_dim=36, seed=int(seed) + 1)
    other.create_model(cache="off")
    second = dam.Model(state_dim=64, seed=seed)
    second.create_model(cache="off")
    assert_array_equal(first.reward_matrix, second.reward_matrix)
    assert first.name == second.name == f"64_8_dam_v2_seed{seed}"
    assert first.get_config()["seed"] == int(seed)
    for left, right in zip(first.transition_matrix, second.transition_matrix):
        assert_array_equal(left.toarray(), right.toarray())

    different = dam.Model(state_dim=64, seed=int(seed) + 1)
    different.create_model(cache="off")
    assert different.name != first.name
    assert np.any(different.reward_matrix != first.reward_matrix)
    for left, right in zip(first.transition_matrix, different.transition_matrix):
        assert_array_equal(left.toarray(), right.toarray())
    after = np.random.get_state()
    assert global_state[0] == after[0]
    assert_array_equal(global_state[1], after[1])
    assert global_state[2:] == after[2:]


def test_dam_default_seed_matches_explicit_zero():
    default = dam.Model(state_dim=16)
    explicit = dam.Model(state_dim=16, seed=0)
    default.create_model(cache="off")
    explicit.create_model(cache="off")
    assert default.seed == 0
    assert default.name == explicit.name
    assert_array_equal(default.reward_matrix, explicit.reward_matrix)


@pytest.mark.parametrize("seed", [-1, True, np.bool_(True), None, 1.5, "0"])
def test_dam_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        dam.Model(seed=seed)
