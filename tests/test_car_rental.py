import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdpforge.core.validation import validate_model
from mdpforge.models.car_rental import Model


def scalar_outcomes(model, cars, request_rate, return_rate):
    request_probs = model._poisson_probabilities(request_rate, model.poisson_cutoff)
    return_probs = model._poisson_probabilities(return_rate, model.poisson_cutoff)
    outcomes = {}
    for requests, p_request in enumerate(request_probs):
        rentals = min(cars, requests)
        for returns, p_return in enumerate(return_probs):
            probability = p_request * p_return
            target = min(model.max_cars, cars - rentals + returns)
            values = outcomes.setdefault(target, [0.0, 0.0])
            values[0] += probability
            values[1] += probability * rentals
    return outcomes


@pytest.mark.parametrize("state_dim, action_dim", [(25, 3), (49, 11)])
def test_car_rental_matches_scalar_outcome_enumeration(state_dim, action_dim):
    model = Model(state_dim=state_dim, action_dim=action_dim)
    model.create_model(cache="off")
    validate_model(model)

    outcomes = [
        [
            scalar_outcomes(
                model,
                cars,
                model.request_rates[location],
                model.return_rates[location],
            )
            for cars in range(model.max_cars + 1)
        ]
        for location in range(2)
    ]
    for cars_1 in range(model.max_cars + 1):
        for cars_2 in range(model.max_cars + 1):
            state = model._encode(cars_1, cars_2)
            for action, requested_move in enumerate(model.action_values):
                lower = -min(cars_2, model.max_cars - cars_1)
                upper = min(cars_1, model.max_cars - cars_2)
                move = min(upper, max(lower, int(requested_move)))
                probabilities = np.zeros(model.state_dim)
                expected_rentals = 0.0
                for next_1, (p_1, rentals_1) in outcomes[0][cars_1 - move].items():
                    for next_2, (p_2, rentals_2) in outcomes[1][cars_2 + move].items():
                        probabilities[model._encode(next_1, next_2)] += p_1 * p_2
                        expected_rentals += rentals_1 * p_2 + rentals_2 * p_1
                assert_array_equal(
                    model.transition_matrix[action].getrow(state).toarray().ravel(),
                    probabilities,
                )
                assert model.reward_matrix[state, action] == (
                    model.rental_reward * expected_rentals
                    - model.movement_cost * abs(move)
                )

    for left, right in zip(model.transition_matrix, model.transition_matrix[1:]):
        for field in ("data", "indices", "indptr"):
            assert not np.shares_memory(getattr(left, field), getattr(right, field))


def test_car_rental_with_no_requests_or_returns_is_deterministic():
    model = Model(state_dim=25, action_dim=11)
    model.request_rates = (0.0, 0.0)
    model.return_rates = (0.0, 0.0)
    model.create_model(cache="off")
    validate_model(model)

    for action, requested_move in enumerate(model.action_values):
        transition = model.transition_matrix[action]
        assert transition.nnz == model.state_dim
        for cars_1 in range(model.max_cars + 1):
            for cars_2 in range(model.max_cars + 1):
                state = model._encode(cars_1, cars_2)
                lower = -min(cars_2, model.max_cars - cars_1)
                upper = min(cars_1, model.max_cars - cars_2)
                move = min(upper, max(lower, int(requested_move)))
                destination = model._encode(cars_1 - move, cars_2 + move)
                assert transition[state, destination] == 1.0
                assert model.reward_matrix[state, action] == -2.0 * abs(move)
