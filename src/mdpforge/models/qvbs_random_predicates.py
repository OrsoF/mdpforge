"""Random Predicates with scalable PPDDL instances and pinned JANI variants.

Reachable valuations and locations define states; enabled choices define actions.
Missing action slots repeat the first choice; stopped states absorb at zero.
Probability objectives pay a signed unit bonus on first reaching the target;
discounting replaces unbounded probabilities and outer verification assertions.
Native rewards retain their scale; minimisation objectives are negated.
Default instances ground the original action schemas over selected constants.
OZ is retained first, including its original UV goal and initial self-predicate.
Goals and initial facts involving omitted objects are removed; reset rules remain.
An explicit variant selects the unmodified published JANI instance.
Sources and parameters: _qvbs/_data/model_manifest.json.
"""

import math

from mdpforge.models._jani.model import JANIModel
from mdpforge.models._qvbs.planning import _count, build_instance

METADATA = {
    "category": "abstract",
    "description": (
        "Random Predicates with configurable object count and original PPDDL rules."
    ),
    "reference": "http://idm-lab.org/wiki/icaps/ipc2006/probabilistic/",
    "model_type": "mdp",
    "source_collection": "qvbs",
    "source_url": "https://github.com/ahartmanns/qcomp/tree/a96cf379885c0a160a552a8fe0de417e2b55b161/benchmarks/mdp/random-predicates",
    "variants": 4,
    "sizes": {
        "small": {
            "requested_state_dim": 64,
            "parameters": {"objects": 1},
        },
        "medium": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": (
                "Two-object instances exceed 10000 states; sizing is not calibrated."
            ),
        },
        "large": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": "Larger object domains require explicit sizing and calibration.",
        },
    },
}

OBJECTIVES = ("goal",)
VARIANTS = (
    "random-predicates.a.jani",
    "random-predicates.b.jani",
    "random-predicates.c.jani",
    "random-predicates.d.jani",
)

TEST_PARAMETERS = {"state_dim": 64, "objects": 1}


class Model(JANIModel):
    """Construct a scaled planning instance or an explicit published JANI variant."""

    collection = "qvbs"
    benchmark = "random-predicates"

    def __init__(
        self,
        state_dim: int = 64,
        *,
        objective: str | None = None,
        variant: str | None = None,
        constants: dict | None = None,
        state_limit: int | None = 1_000_000,
        objects: int | None = None,
    ):
        """objective: None or OBJECTIVES; variant: None or VARIANTS; selects source rules.
        constants: None or source constant overrides; state_limit: positive int or None.
        state_dim: positive size target; selects object count for generated instances.
        objects: 1..10 or None for automatic; cannot accompany variant."""
        self._planning_parameters = None
        self._planning_config = None
        if variant is None:
            if type(state_dim) is not int or state_dim < 1:
                raise ValueError("state_dim must be a positive integer")
            if objects is None:
                objects = min(
                    range(1, 11),
                    key=lambda n: abs(
                        (3 * n**2 + 2 * n + 1) * math.log(2) - math.log(state_dim)
                    ),
                )
            self._planning_parameters = {"objects": _count(objects, "objects", 1, 10)}
        elif objects is not None:
            raise ValueError("variant cannot be combined with objects")
        super().__init__(
            state_dim,
            objective=objective,
            variant="random-predicates.a.jani" if variant is None else variant,
            constants=constants,
            state_limit=state_limit,
        )

    def _prepare_source(self, document, reward_spec):
        """Ground the original rules over a deterministic subset of constants."""
        if self._planning_parameters is None:
            return document, reward_spec
        document, goal, self._planning_config = build_instance(
            self.benchmark, self._planning_parameters
        )
        return document, reward_spec | {"reach": goal}

    def get_config(self):
        """Include selected objects and source hashes in cache identity."""
        return super().get_config() | {"planning_instance": self._planning_config}
