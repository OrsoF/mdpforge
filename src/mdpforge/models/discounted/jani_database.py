"""Uniformized database reliability with a discounted first-failure objective.

The pinned JANI source repairs components independently, unlike its prose
description of a shared repairman. Each failure races two equal-rate clocks,
selecting exponential repair rates 1 and 0.5. States count failed components
by repair mode within a group, then group identical processor/controller and
disk groups by symmetry. Reaching the source rare-event predicate pays +1
and absorbs; this is a first-failure adaptation of its steady-state property.
"""

import math
from collections import Counter
from itertools import combinations_with_replacement, product

from mdpforge.models.discounted._jani.uniformization import (
    UniformizedModel,
    load_timed_source,
    positive_integer,
)

METADATA = {
    "category": "service_operations",
    "description": "Uniformized database reliability, aggregated by symmetry, with a first-system-failure reward.",
    "reference": "Villen-Altamirano and Villen-Altamirano (2011), The rare event simulation method RESTART; pinned JANI Database dynamics.",
    "model_type": "mdp",
    "source_model_type": "sta",
    "source_collection": "jani-models",
    "source_url": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/Database",
    "sizes": {
        "tiny": {"requested_state_dim": 421, "parameters": {"redundancy": 2}},
        "small": {"requested_state_dim": 58213, "parameters": {"redundancy": 3}},
        "medium": {"requested_state_dim": 3578576, "parameters": {"redundancy": 4}},
        "large": {"requested_state_dim": 118605601, "parameters": {"redundancy": 5}},
    },
}

TEST_PARAMETERS = {"state_dim": 421, "redundancy": 2}


def _state_count(redundancy):
    """Count symmetric nonfailure states and one absorbing failure state."""
    categories = redundancy * (redundancy + 1) // 2
    return math.comb(categories + 3, 4) * math.comb(categories + 5, 6) + 1


class Model(UniformizedModel):
    """Preserve component races and repair modes before the first system failure."""

    adaptation = {
        "method": "uniformization with component and identical-group symmetry lumping",
        "objective": "discounted +1 on first system failure, replacing source steady-state occupancy",
        "repairs": "independent repairs as encoded by JANI, not a shared repairman",
        "clock_initialization": "fresh exponential failure waits; all components operational",
        "scaling": "published redundancy R=2,3,4,5; four processor/controller groups and six disk groups",
    }

    def __init__(self, state_dim: int = 421, *, redundancy: int | None = None):
        """state_dim: positive target; redundancy: integer 2..5 or None, nearest published size."""
        positive_integer(state_dim, "state_dim")
        if redundancy is None:
            redundancy = min(
                range(2, 6),
                key=lambda r: (abs(math.log(_state_count(r) / state_dim)), r),
            )
        self.redundancy = positive_integer(redundancy, "redundancy", 2)
        if self.redundancy > 5:
            raise ValueError("redundancy must be one of the bundled values 2, 3, 4, 5")
        document, source = load_timed_source(f"Database/database_R{redundancy}.jani")
        automata = {item["name"]: item for item in document["automata"]}
        rates = []
        for name in ("Processor11", "Disk11"):
            distributions = [
                assignment["value"]
                for edge in automata[name]["edges"]
                for destination in edge["destinations"]
                for assignment in destination.get("assignments", [])
                if isinstance(assignment["value"], dict)
                and assignment["value"].get("distribution") == "Exponential"
            ]
            rates.append(min(float(item["args"][0]) for item in distributions))
        self.failure_rates = tuple(rates)
        self.categories = tuple(
            (first, second)
            for first in range(redundancy)
            for second in range(redundancy - first)
        )
        self.category_indices = {
            value: index for index, value in enumerate(self.categories)
        }
        self.failed_state = ((), ())
        rate = sum(
            groups
            * max(
                2 * failure * (capacity - first - second) + first + 0.5 * second
                for first, second in self.categories
            )
            for groups, capacity, failure in zip(
                (4, 6), (redundancy, redundancy + 2), self.failure_rates
            )
        )
        super().__init__(
            state_dim,
            _state_count(redundancy),
            1,
            ((0,) * 4, (0,) * 6),
            rate,
            {
                "redundancy": redundancy,
                "failure_clock_rates": self.failure_rates,
                "repair_rates": (1.0, 0.5),
                "objective": "first_failure",
            },
            [source],
        )

    def _states(self):
        """Enumerate multisets of equivalent component groups and a failure state."""
        for processors, disks in product(
            combinations_with_replacement(range(len(self.categories)), 4),
            combinations_with_replacement(range(len(self.categories)), 6),
        ):
            yield processors, disks
        yield self.failed_state

    def _terminal(self, state):
        """Stop when any source component group reaches its failure threshold."""
        return state == self.failed_state

    def _rate_choices(self, state):
        """Aggregate failure races and independent repairs over equivalent groups."""
        events = []
        for kind, groups in enumerate(state):
            capacity = self.redundancy + (2 if kind else 0)
            failure = self.failure_rates[kind]
            for category, multiplicity in Counter(groups).items():
                first, second = self.categories[category]
                changes = [
                    ((first + 1, second), (capacity - first - second) * failure),
                    ((first, second + 1), (capacity - first - second) * failure),
                ]
                if first:
                    changes.append(((first - 1, second), float(first)))
                if second:
                    changes.append(((first, second - 1), 0.5 * second))
                for counts, rate in changes:
                    failed = sum(counts) == self.redundancy
                    if failed:
                        successor = self.failed_state
                    else:
                        updated = list(groups)
                        updated[updated.index(category)] = self.category_indices[counts]
                        successor = list(state)
                        successor[kind] = tuple(sorted(updated))
                        successor = tuple(successor)
                    events.append((successor, multiplicity * rate, float(failed)))
        yield events
