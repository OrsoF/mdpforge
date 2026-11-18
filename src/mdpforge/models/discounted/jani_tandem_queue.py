"""Uniformized tandem queues with a discounted first-overflow reward.

States are (q1, q2); arrivals and services retain their source rates.
A full second queue pays +1 once; emptying stops at zero reward.
Fresh exponential waiting times replace the zero-initialized source clocks.
"""

import math
from itertools import product

from mdpforge.models.discounted._jani.uniformization import (
    UniformizedModel,
    load_timed_source,
    positive_integer,
)

METADATA = {
    "category": "service_operations",
    "description": "Uniformized tandem queues: reach a full second queue before it empties.",
    "reference": "Garvels (2000), The splitting method in rare event simulation; pinned JANI TandemQueue source.",
    "model_type": "mdp",
    "source_model_type": "sta",
    "source_collection": "jani-models",
    "source_url": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/TandemQueue",
    "sizes": {
        "tiny": {"requested_state_dim": 81, "parameters": {"capacity": 8}},
        "small": {"requested_state_dim": 1024, "parameters": {"capacity": 31}},
        "medium": {"requested_state_dim": 10000, "parameters": {"capacity": 99}},
        "large": {"requested_state_dim": 99856, "parameters": {"capacity": 315}},
    },
}

TEST_PARAMETERS = {"state_dim": 9, "capacity": 2}


class Model(UniformizedModel):
    """Preserve exponential queue events using a single uniformized action."""

    adaptation = {
        "method": "uniformization of exponential arrivals and services",
        "objective": "discounted +1 on first reaching q2=capacity before q2=0",
        "clock_initialization": "fresh exponential waits, including at time zero",
        "scaling": "capacity varies; source event rates are unchanged",
    }

    def __init__(self, state_dim: int = 81, *, capacity: int | None = None):
        """state_dim: positive target; capacity: integer >=2 or None, inferred."""
        positive_integer(state_dim, "state_dim")
        self.capacity = positive_integer(
            max(2, math.isqrt(state_dim) - 1) if capacity is None else capacity,
            "capacity",
            2,
        )
        document, source = load_timed_source("TandemQueue/tandem_queue-iosa.jani")
        constants = {item["name"]: item["value"] for item in document["constants"]}
        self.rates = tuple(float(constants[name]) for name in ("lambda", "mu1", "mu2"))
        super().__init__(
            state_dim,
            (self.capacity + 1) ** 2,
            1,
            (0, 1),
            sum(self.rates),
            {"capacity": self.capacity, "rates": self.rates},
            [source],
        )

    def _states(self):
        """List all finite queue occupancies."""
        return product(range(self.capacity + 1), repeat=2)

    def _terminal(self, state):
        """Stop at overflow or emptying of the second queue."""
        return state[1] in (0, self.capacity)

    def _rate_choices(self, state):
        """Race arrivals, transfers and departures at their source rates."""
        q1, q2 = state
        arrival, service1, service2 = self.rates
        events = [((min(q1 + 1, self.capacity), q2), arrival, 0.0)]
        if q1:
            events.append(((q1 - 1, q2 + 1), service1, float(q2 + 1 == self.capacity)))
        if q2:
            events.append(((q1, q2 - 1), service2, 0.0))
        yield events
