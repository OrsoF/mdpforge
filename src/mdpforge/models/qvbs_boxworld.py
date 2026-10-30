"""Boxworld with scalable PPDDL instances and pinned full JANI variants.

Reachable valuations and locations define states; enabled choices define actions.
Missing action slots repeat the first choice; stopped states absorb at zero.
Probability objectives pay a signed unit bonus on first reaching the target;
discounting replaces unbounded probabilities and outer verification assertions.
Native rewards retain their scale; minimisation objectives are negated.
By default, ground the original five-city domain with one box and one truck.
Object subsets retain the original drive probabilities and load/unload rules.
An explicit variant selects the unmodified published JANI instance.
Sources and parameters: _qvbs/_data/model_manifest.json.
"""

import math

from mdpforge.models._jani.model import JANIModel
from mdpforge.models._qvbs.planning import _count, build_instance

METADATA = {
    "category": "service_operations",
    "description": (
        "Boxworld with configurable boxes and vehicles from original PPDDL rules."
    ),
    "reference": "https://doi.org/10.1609/aimag.v21i2.1506",
    "model_type": "mdp",
    "source_collection": "qvbs",
    "source_url": "https://github.com/ahartmanns/qcomp/tree/a96cf379885c0a160a552a8fe0de417e2b55b161/benchmarks/mdp/boxworld",
    "variants": 5,
    "sizes": {
        "small": {
            "requested_state_dim": 30,
            "parameters": {"boxes": 1, "trucks": 1, "planes": 0},
        },
        "medium": {
            "requested_state_dim": 180,
            "parameters": {"boxes": 2, "trucks": 1, "planes": 0},
        },
        "large": {
            "requested_state_dim": 1080,
            "parameters": {"boxes": 3, "trucks": 1, "planes": 0},
        },
    },
}

OBJECTIVES = ("goal",)
VARIANTS = (
    "boxworld.10-10.jani",
    "boxworld.10-5.jani",
    "boxworld.15-10.jani",
    "boxworld.15-15.jani",
    "boxworld.20-20.jani",
)

TEST_PARAMETERS = {"state_dim": 30, "boxes": 1, "trucks": 1, "planes": 0}


class Model(JANIModel):
    """Construct the bundled boxworld model with an explicit discounted objective."""

    collection = "qvbs"
    benchmark = "boxworld"

    def __init__(
        self,
        state_dim: int = 30,
        *,
        objective: str | None = None,
        variant: str | None = None,
        constants: dict | None = None,
        state_limit: int | None = 1_000_000,
        boxes: int | None = None,
        trucks: int | None = None,
        planes: int | None = None,
    ):
        """objective: None or OBJECTIVES; variant: None or VARIANTS; selects source rules.
        constants: None or source constant overrides; state_limit: positive int or None.
        state_dim: positive size target; selects box count for generated instances.
        boxes: 1..10 or None for automatic; trucks: 1..4 or None for one.
        planes: 0..2 or None for zero; object counts cannot accompany variant."""
        self._planning_parameters = None
        self._planning_config = None
        if variant is None:
            if type(state_dim) is not int or state_dim < 1:
                raise ValueError("state_dim must be a positive integer")
            trucks = _count(1 if trucks is None else trucks, "trucks", 1, 4)
            planes = _count(0 if planes is None else planes, "planes", 0, 2)
            if boxes is None:
                boxes = min(
                    range(1, 11),
                    key=lambda n: abs(
                        math.log(5**trucks * 2**planes * (5 + trucks + planes)**n)
                        - math.log(state_dim)
                    ),
                )
            self._planning_parameters = {
                "boxes": _count(boxes, "boxes", 1, 10),
                "trucks": trucks,
                "planes": planes,
            }
        elif any(value is not None for value in (boxes, trucks, planes)):
            raise ValueError("variant cannot be combined with boxes, trucks or planes")
        super().__init__(
            state_dim,
            objective=objective,
            variant="boxworld.10-5.jani" if variant is None else variant,
            constants=constants,
            state_limit=state_limit,
        )

    def _prepare_source(self, document, reward_spec):
        """Ground a smaller object instance while preserving the domain rules."""
        if self._planning_parameters is None:
            return document, reward_spec
        document, goal, self._planning_config = build_instance(
            self.benchmark, self._planning_parameters
        )
        return document, reward_spec | {"reach": goal}

    def get_config(self):
        """Include object selection and original PPDDL hashes in cache identity."""
        return super().get_config() | {"planning_instance": self._planning_config}
