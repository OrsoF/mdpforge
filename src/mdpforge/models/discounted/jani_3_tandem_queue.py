"""Finite Erlang phases and uniformization of the triple tandem queue.

Each queue code is 0 when idle, otherwise 1+shape*(occupancy-1)+phase.
Service phase rates are 1/beta, using the bundled README's scale convention.
Transfers to full queues are lost; each tick pays the q3>=threshold indicator.
"""

from itertools import product

from mdpforge.models.discounted._jani.uniformization import (
    UniformizedModel,
    load_timed_source,
    positive_integer,
)

METADATA = {
    "category": "service_operations",
    "model_type": "mdp",
    "source_model_type": "sta",
    "source_collection": "jani-models",
    "source_url": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/3TandemQueue",
    "sizes": {
        "tiny": {"requested_state_dim": 2197, "parameters": {"capacity": 6}},
        "small": {"requested_state_dim": 15625, "parameters": {"capacity": 12}},
        "medium": {"requested_state_dim": 68921, "parameters": {"capacity": 20}},
        "large": {"requested_state_dim": 300763, "parameters": {"capacity": 33}},
    },
}

TEST_PARAMETERS = {"state_dim": 27, "capacity": 1, "threshold": 1}


class Model(UniformizedModel):
    """Keep residual service information through finite Erlang phase counters."""

    adaptation = {
        "method": "Erlang phase expansion followed by uniformization",
        "objective": "+1 per uniformized tick with q3>=threshold; no terminal absorption",
        "erlang_convention": "shape alpha and scale beta, following the bundled README",
        "clock_initialization": "fresh arrival and service waits when activated",
        "scaling": "capacity varies; default threshold=capacity-5, at least 1",
    }

    def __init__(
        self,
        state_dim: int = 35937,
        *,
        capacity: int | None = None,
        shape: int = 2,
        threshold: int | None = None,
    ):
        """state_dim: positive target; capacity/shape: positive int; threshold: 1..capacity or None."""
        positive_integer(state_dim, "state_dim")
        self.shape = positive_integer(shape, "shape")
        inferred = max(6, round((state_dim ** (1 / 3) - 1) / self.shape))
        self.capacity = positive_integer(
            inferred if capacity is None else capacity, "capacity"
        )
        self.threshold = positive_integer(
            max(1, self.capacity - 5) if threshold is None else threshold, "threshold"
        )
        if self.threshold > self.capacity:
            raise ValueError("threshold must not exceed capacity")
        document, source = load_timed_source("3TandemQueue/3tandem_queue-iosa.jani")
        _, convention = load_timed_source("3TandemQueue/README.md")
        constants = {item["name"]: item["value"] for item in document["constants"]}
        self.service_rates = tuple(
            1.0 / constants[name] for name in ("beta1", "beta2", "beta3")
        )
        self.queue_codes = 1 + self.shape * self.capacity
        super().__init__(
            state_dim,
            self.queue_codes**3,
            1,
            (0, 0, 0),
            1 + sum(self.service_rates),
            {
                "capacity": self.capacity,
                "shape": self.shape,
                "threshold": self.threshold,
                "arrival_rate": 1.0,
                "service_phase_rates": self.service_rates,
            },
            [source, convention],
        )

    def _states(self):
        """List occupancy-phase codes for all three queues."""
        return product(range(self.queue_codes), repeat=3)

    def _state_reward(self, state):
        """Measure threshold occupancy in the last queue at each tick."""
        return float(state[2] > self.shape * (self.threshold - 1))

    def _rate_choices(self, state):
        """Advance service phases and pass completed jobs downstream."""
        successor = list(state)
        successor[0] = self._arrive(state[0])
        events = [(tuple(successor), 1.0, 0.0)]
        for queue, code in enumerate(state):
            if code == 0:
                continue
            successor = list(state)
            occupancy, phase = divmod(code - 1, self.shape)
            if phase + 1 < self.shape:
                successor[queue] += 1
            else:
                successor[queue] = (
                    0 if occupancy == 0 else 1 + self.shape * (occupancy - 1)
                )
                if queue < 2:
                    successor[queue + 1] = self._arrive(state[queue + 1])
            events.append((tuple(successor), self.service_rates[queue], 0.0))
        yield events

    def _arrive(self, code):
        """Start service in an idle queue or append a job when space remains."""
        if code == 0:
            return 1
        return (
            code + self.shape if (code - 1) // self.shape + 1 < self.capacity else code
        )
