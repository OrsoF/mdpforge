"""Native PRISM csma command semantics with discounted completion costs.

States are reachable integer valuations; actions are local choice offsets.
Disabled slots repeat a valid choice; terminal transitions are not removed.
Costs are negated and zeroed on completion, with no terminal bonus or scaling.
state_dim selects the closest bundled configuration unless parameters fix it.
Original authors and CC-BY 4.0 source attribution: _data/prism/manifest.json.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._prism import build_prism, configure_prism

METADATA = {
    "category": "service_operations",
    "description": (
        "PRISM CSMA/CD with native bus synchronisation, station clocks and "
        "exponential backoff."
    ),
    "reference": (
        "Kwiatkowska, Norman and Parker, The PRISM Benchmark Suite, QEST "
        "2012; CC-BY 4.0: "
        "https://github.com/prismmodelchecker/prism-benchmarks"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 1038,
            "parameters": {"stations": 2, "backoff": 2},
        },
        "medium": {
            "requested_state_dim": 533744,
            "parameters": {"stations": 2, "backoff": 6},
        },
        "large": {
            "requested_state_dim": 761962,
            "parameters": {"stations": 4, "backoff": 2},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 1038, "stations": 2, "backoff": 2}


class Model(MDP):
    """Construct the published reachable csma MDP in Python."""

    def __init__(
        self,
        state_dim: int = 1038,
        *,
        stations: int = 2,
        backoff: int | None = None,
        reward: str = "time",
    ):
        """stations: 2, 3 or 4; stations; default 2.
        backoff: None selects by state_dim, or 2, 4, 6; PRISM K; default None.
        Only K=2 is bundled for three or four stations.
        reward: time; native elapsed time cost."""
        if type(stations) is not int or stations not in (2, 3, 4):
            raise ValueError("stations must be 2, 3 or 4")
        if backoff is not None and (
            type(backoff) is not int or backoff not in (2, 4, 6)
        ):
            raise ValueError("backoff must be None, 2, 4 or 6")
        configure_prism(
            self,
            "csma",
            state_dim,
            {"stations": stations, "backoff": backoff},
            reward,
        )

    def _build_model(self):
        build_prism(self)
