"""Invariants for the eight *reduced* planning/protocol benchmark recipes."""

import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge.models.boxworld import Model as Boxworld
from mdpforge.models.deep_sea_treasure import Model as DeepSeaTreasure
from mdpforge.models.drone_energy import Model as DroneEnergy
from mdpforge.models.firewire import Model as Firewire
from mdpforge.models.search_and_rescue import Model as SearchAndRescue
from mdpforge.models.traffic_control import Model as TrafficControl
from mdpforge.models.wlan import Model as WLAN
from mdpforge.models.zenotravel import Model as ZenoTravel


NEW_MODELS = (
    "drone_energy",
    "traffic_control",
    "deep_sea_treasure",
    "search_and_rescue",
    "zenotravel",
    "boxworld",
    "wlan",
    "firewire",
)


@pytest.mark.parametrize("name", NEW_MODELS)
@pytest.mark.parametrize("size", ("small", "medium", "large"))
def test_presets_are_stochastic_and_finite(name, size):
    module = importlib.import_module(f"mdpforge.models.{name}")
    preset = module.METADATA["sizes"][size]
    model = module.Model(
        state_dim=preset["requested_state_dim"], **preset["parameters"]
    )
    model.create_model(save=False)
    model.test_model()
    assert model.requested_state_dim == preset["requested_state_dim"]
    assert model.state_dim > 0 and model.action_dim > 0
    assert all(matrix.nnz < 25 * model.state_dim for matrix in model.transition_matrix)
    assert all(np.all(np.isfinite(matrix.data)) for matrix in model.transition_matrix)
    assert np.isfinite(model.reward_matrix).all()
    assert model.name


@pytest.mark.parametrize("name", NEW_MODELS)
def test_invalid_requested_dimension(name):
    recipe = importlib.import_module(f"mdpforge.models.{name}").Model
    with pytest.raises(ValueError, match="state_dim"):
        recipe(state_dim=0)
    with pytest.raises(ValueError, match="state_dim"):
        recipe(state_dim=True)


def test_drone_battery_wind_charging_and_crash():
    model = DroneEnergy(state_dim=50, wind_probability=0.25, stall_probability=0.15)
    model.create_model(save=False)
    initial = model._encode(0, 1, 0)
    assert_allclose(
        model.transition_matrix[0][initial, model._encode(1, 0, 0)], 0.95 * 0.75
    )
    assert_allclose(
        model.transition_matrix[0][initial, model._encode(0, 0, 1)], 0.05 * 0.25
    )
    out = model._encode(1, 0, 0)
    assert_allclose(model.transition_matrix[0][out, model.crashed], 1)
    charge = model._encode(0, 0, 0)
    assert_allclose(model.transition_matrix[2][charge, model._encode(0, 2, 0)], 0.75)
    assert_allclose(model.transition_matrix[1][model.landed, model.landed], 1)


def test_traffic_arrivals_and_phase_switch():
    model = TrafficControl(state_dim=50, arrival_north=0.45, arrival_east=0.35)
    model.create_model(save=False)
    state = model._encode(1, 0, 0)
    assert_allclose(
        model.transition_matrix[0][state, model._encode(1, 1, 0)], 0.45 * 0.35
    )
    assert_allclose(
        model.transition_matrix[1][state, model._encode(1, 0, 1)],
        (1 - 0.45) * (1 - 0.35),
    )
    assert model.reward_matrix[state, 1] < model.reward_matrix[state, 0]


def test_treasure_collect_surface_and_drowning():
    model = DeepSeaTreasure(state_dim=74)
    model.create_model(save=False)
    target = model._encode(1, 2, 3)
    assert_allclose(model.transition_matrix[4][target, model.treasure], 1)
    assert model.reward_matrix[target, 4] > 0
    surface = model._encode(0, 1, 0)
    full = model._encode(0, 1, model.oxygen_capacity)
    assert_allclose(model.transition_matrix[4][surface, full], 1)
    no_air = model._encode(1, 1, 0)
    assert_allclose(model.transition_matrix[0][no_air, model.drowned], 1)


def test_rescue_site_survey_pickup_and_evacuation():
    model = SearchAndRescue(state_dim=74, per_step_loss=0.05)
    model.create_model(save=False)
    scout = model._encode(1, 0, 0, 2)
    safe = model._encode(1, 1, 0, 1)
    assert_allclose(model.transition_matrix[2][scout, safe], 0.65 * 0.95)
    assert_allclose(model.transition_matrix[2][scout, model.lost], 0.05)
    pickup = model._encode(1, 1, 0, 1)
    carrying = model._encode(1, 1, 1, 0)
    assert_allclose(model.transition_matrix[3][pickup, carrying], 0.85 * 0.95)
    assert_allclose(model.transition_matrix[4][carrying, model.rescued], 0.95)
    assert_allclose(model.transition_matrix[0][model.lost, model.lost], 1)


def test_zenotravel_board_refuel_fly_deliver():
    model = ZenoTravel(state_dim=58, flight_failure=0.12, accident_probability=0.015)
    model.create_model(save=False)
    state = model._encode(0, 1, 0)
    assert_allclose(model.transition_matrix[0][state, model._encode(0, 1, 1)], 1)
    flying = model._encode(0, 1, 1)
    assert_allclose(
        model.transition_matrix[1][flying, model._encode(1, 0, 1)],
        0.985 * 0.88,
    )
    refuel = model._encode(0, 2, 1)
    assert_allclose(
        model.transition_matrix[3][refuel, model._encode(0, model.fuel_capacity, 1)], 1
    )
    arrive = model._encode(model.cities - 1, 2, 1)
    assert_allclose(model.transition_matrix[4][arrive, model.delivered], 1)


def test_boxworld_load_deliver_and_loss():
    model = Boxworld(state_dim=52, traffic_delay=0.2, cargo_damage=0.025)
    model.create_model(save=False)
    load = model._encode(0, 0, 0)
    assert_allclose(model.transition_matrix[3][load, model._encode(0, 0, 1)], 1)
    loaded = model._encode(0, 0, 1)
    assert_allclose(model.transition_matrix[0][loaded, model.damaged], 0.025)
    assert_allclose(
        model.transition_matrix[0][loaded, model._encode(1, 0, 1)], 0.975 * 0.8
    )
    delivered = model._encode(model.locations - 1, 0, 1)
    assert_allclose(model.transition_matrix[4][delivered, model.delivered], 1)


def test_wlan_collision_and_independent_transmission():
    model = WLAN(state_dim=66, external_interference=0)
    model.create_model(save=False)
    ready = model._encode(0, 0, 0, 0)
    collision = model._encode(1, 1, 1, 0)
    assert_allclose(model.transition_matrix[0][ready, collision], 0.65**2)
    assert_allclose(
        model.transition_matrix[0][ready, model._encode(0, 0, 0, 1)], 0.65 * 0.35
    )
    assert_allclose(
        model.transition_matrix[0][ready, model._encode(0, 0, 0, 2)], 0.65 * 0.35
    )
    limit = model._encode(0, 0, model.max_collisions, 0)
    assert model.transition_matrix[0][limit, model.failure] > 0
    assert_allclose(model.transition_matrix[2][model.success, model.success], 1)


def test_firewire_timer_tie_and_root_election():
    model = Firewire(state_dim=74, tie_probability=0.2)
    model.create_model(save=False)
    start = model._encode(0, 0, 0, 0)
    assert_allclose(
        model.transition_matrix[0][start, model._encode(1, model.delay, 0, 1)], 0.8
    )
    assert_allclose(model.transition_matrix[0][start, model._encode(1, 1, 0, 1)], 0.2)
    expired = model._encode(0, 1, 0, 1)
    assert_allclose(model.transition_matrix[2][expired, model.elected], 1)
    final_tie = model._encode(0, 0, model.max_rounds, 1)
    assert_allclose(model.transition_matrix[2][final_tie, model.deadline], 1)


def test_catalogue_contains_eight_new_names():
    from mdpforge import list_models

    entries = {entry["name"]: entry for entry in list_models()}
    assert set(NEW_MODELS) <= entries.keys()
    assert entries["wlan"]["category"] == "communication_protocols"
    assert entries["traffic_control"]["category"] == "service_operations"
