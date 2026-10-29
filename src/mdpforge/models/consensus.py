"""Native PRISM consensus command semantics with discounted completion costs.

States are reachable integer valuations; actions are local choice offsets.
Disabled slots repeat a valid choice; terminal transitions are not removed.
Costs are negated and zeroed on completion, with no terminal bonus or scaling.
state_dim selects the closest bundled configuration unless parameters fix it.
Original authors and CC-BY 4.0 source attribution: _prism_data/manifest.json.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._prism import build_prism, configure_prism

METADATA = {
    "category": "abstract",
    "description": (
        "PRISM shared-coin consensus with separate flip/write/check phases "
        "and native step cost."
    ),
    "reference": (
        "Kwiatkowska, Norman and Parker, The PRISM Benchmark Suite, QEST "
        "2012; CC-BY 4.0: "
        "https://github.com/prismmodelchecker/prism-benchmarks"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 1040,
            "parameters": {"processes": 2, "bound": 8},
        },
        "medium": {
            "requested_state_dim": 345088,
            "parameters": {"processes": 4, "bound": 4},
        },
        "large": {
            "requested_state_dim": 43136,
            "parameters": {"processes": 4, "bound": 4},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 272, "processes": 2, "bound": 2}


class Model(MDP):
    """Construct the published reachable consensus MDP in Python."""

    def __init__(
        self,
        state_dim: int = 272,
        *,
        processes: int = 2,
        bound: int | None = None,
        reward: str = "steps",
    ):
        """processes: 2 or 4; concurrent processes; default 2.
        bound: None selects by state_dim, or 2, 4, 8, 16; default None.
        Four processes support only K=2 or 4; reward: steps, native step cost."""
        if type(processes) is not int or processes not in (2, 4):
            raise ValueError("processes must be 2 or 4")
        if bound is not None and (type(bound) is not int or bound not in (2, 4, 8, 16)):
            raise ValueError("bound must be None, 2, 4, 8 or 16")
        configure_prism(
            self,
            "consensus",
            state_dim,
            {"processes": processes, "bound": bound},
            reward,
        )

    def _build_model(self):
        build_prism(self)
