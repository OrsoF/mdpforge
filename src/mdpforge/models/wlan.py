"""Native PRISM wlan command semantics with discounted completion costs.

States are reachable integer valuations; actions are local choice offsets.
Disabled slots repeat a valid choice; terminal transitions are not removed.
Costs are negated and zeroed on completion, with no terminal bonus or scaling.
state_dim selects the closest bundled configuration unless parameters fix it.
Original authors and CC-BY 4.0 source attribution: _prism_data/manifest.json.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._prism import build_prism, configure_prism

METADATA = {
    "category": "service_operations",
    "description": (
        "PRISM WLAN with native DIFS/SIFS, acknowledgements, medium and random backoff."
    ),
    "reference": (
        "Kwiatkowska, Norman and Parker, The PRISM Benchmark Suite, QEST "
        "2012; CC-BY 4.0: "
        "https://github.com/prismmodelchecker/prism-benchmarks"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 2954,
            "parameters": {"backoff": 0},
        },
        "medium": {
            "requested_state_dim": 8625,
            "parameters": {"backoff": 1},
        },
        "large": {
            "requested_state_dim": 28480,
            "parameters": {"backoff": 2},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 2954, "backoff": 0}


class Model(MDP):
    """Construct the published reachable wlan MDP in Python."""

    def __init__(
        self,
        state_dim: int = 2954,
        *,
        backoff: int | None = None,
        reward: str = "time",
    ):
        """backoff: None selects by state_dim, or 0..6; native BOFF; default None.
        reward: time, cost or collisions; native cost; default time.
        COL is fixed at 0.
        """
        if backoff is not None and (type(backoff) is not int or not 0 <= backoff <= 6):
            raise ValueError("backoff must be None or an integer in [0, 6]")
        configure_prism(
            self,
            "wlan",
            state_dim,
            {"backoff": backoff},
            reward,
        )

    def _build_model(self):
        build_prism(self)
