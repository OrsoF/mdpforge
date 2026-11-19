"""Uniformized breakdown queue with exact source-count aggregation.

States are (buffer, server_on, active_type1, active_type2).
Identical exponential sources are counted instead of individually labelled.
The source's first-overflow event pays +1; emptying stops at zero reward.
Source and server waiting times are sampled afresh at initialization.
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
    "source_url": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/QueueWithBreakdowns",
    "sizes": {
        "tiny": {"requested_state_dim": 2952, "parameters": {"capacity": 40}},
        "small": {"requested_state_dim": 10152, "parameters": {"capacity": 140}},
        "medium": {"requested_state_dim": 36072, "parameters": {"capacity": 500}},
        "large": {"requested_state_dim": 144072, "parameters": {"capacity": 2000}},
    },
}

TEST_PARAMETERS = {"state_dim": 24, "capacity": 2, "sources_per_type": (1, 1)}


class Model(UniformizedModel):
    """Use memorylessness to combine identical on/off source populations."""

    adaptation = {
        "method": "exponential uniformization with source-count lumping",
        "objective": "discounted +1 on first full buffer before reset on emptying",
        "clock_initialization": "fresh exponential waits, retaining source on/off states",
        "scaling": "buffer capacity varies; rates and default source populations are retained",
    }

    def __init__(
        self,
        state_dim: int = 2952,
        *,
        capacity: int | None = None,
        sources_per_type: tuple[int, int] = (5, 5),
    ):
        """state_dim: positive target; capacity: int>=2 or None; sources_per_type: two positive ints."""
        positive_integer(state_dim, "state_dim")
        if not isinstance(sources_per_type, tuple) or len(sources_per_type) != 2:
            raise ValueError(
                "sources_per_type must be a tuple of two positive integers"
            )
        self.populations = tuple(
            positive_integer(n, "sources_per_type") for n in sources_per_type
        )
        multiplier = 2 * (self.populations[0] + 1) * (self.populations[1] + 1)
        self.capacity = positive_integer(
            max(2, round(state_dim / multiplier) - 1) if capacity is None else capacity,
            "capacity",
            2,
        )
        document, source = load_timed_source(
            "QueueWithBreakdowns/queue_with_breakdowns-iosa.jani"
        )
        self.rates = {
            item["name"]: item["value"]
            for item in document["constants"]
            if item["name"] != "K"
        }
        rate = self.rates["mu"] + max(self.rates["gama"], self.rates["delta"])
        for kind, count in enumerate(self.populations, 1):
            rate += count * (
                self.rates[f"lambda{kind}"]
                + max(self.rates[f"alpha{kind}"], self.rates[f"beta{kind}"])
            )
        super().__init__(
            state_dim,
            multiplier * (self.capacity + 1),
            1,
            (1, 0, 0, 1),
            rate,
            {
                "capacity": self.capacity,
                "sources_per_type": self.populations,
                "rates": self.rates,
            },
            [source],
        )

    def _states(self):
        """List finite occupancies and aggregated on/off source counts."""
        return product(
            range(self.capacity + 1),
            range(2),
            range(self.populations[0] + 1),
            range(self.populations[1] + 1),
        )

    def _terminal(self, state):
        """Stop when the source overflow or empty-buffer event occurs."""
        return state[0] in (0, self.capacity)

    def _rate_choices(self, state):
        """Race active arrivals, repairs, failures and enabled service."""
        buffer, server, first, second = state
        arrival = first * self.rates["lambda1"] + second * self.rates["lambda2"]
        events = [
            (
                (buffer + 1, server, first, second),
                arrival,
                float(buffer + 1 == self.capacity),
            )
        ]
        events.append(
            (
                (buffer, 1 - server, first, second),
                self.rates["gama" if server else "delta"],
                0.0,
            )
        )
        if server:
            events.append(((buffer - 1, server, first, second), self.rates["mu"], 0.0))
        for position, (active, count) in enumerate(
            zip((first, second), self.populations), 2
        ):
            kind = position - 1
            if active:
                successor = list(state)
                successor[position] -= 1
                events.append(
                    (tuple(successor), active * self.rates[f"beta{kind}"], 0.0)
                )
            if active < count:
                successor = list(state)
                successor[position] += 1
                events.append(
                    (
                        tuple(successor),
                        (count - active) * self.rates[f"alpha{kind}"],
                        0.0,
                    )
                )
        yield events
