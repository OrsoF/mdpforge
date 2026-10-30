"""Native PRISM zeroconf command semantics with discounted completion costs.

States are reachable integer valuations; actions are local choice offsets.
Disabled slots repeat a valid choice; terminal transitions are not removed.
Costs are negated and zeroed on completion, with no terminal bonus or scaling.
state_dim selects the closest bundled configuration unless parameters fix it.
Original authors and CC-BY 4.0 source attribution: _prism/_data/manifest.json.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._prism.reader import build_prism, configure_prism

METADATA = {
    "category": "service_operations",
    "description": (
        "PRISM Zeroconf with message buffers, digital clocks, defence and "
        "reset semantics."
    ),
    "reference": (
        "Kwiatkowska, Norman and Parker, The PRISM Benchmark Suite, QEST "
        "2012; CC-BY 4.0: "
        "https://github.com/prismmodelchecker/prism-benchmarks"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 1924,
            "parameters": {"probes": 8, "reset": True},
        },
        "medium": {
            "requested_state_dim": 89586,
            "parameters": {"probes": 2, "reset": False},
        },
        "large": {
            "requested_state_dim": 307768,
            "parameters": {"probes": 4, "reset": False},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 670, "probes": 2, "reset": True}


class Model(MDP):
    """Construct the published reachable zeroconf MDP in Python."""

    def __init__(
        self,
        state_dim: int = 670,
        *,
        probes: int | None = None,
        existing_hosts: int = 1000,
        reset: bool = True,
        reward: str = "time",
    ):
        """probes: None selects by state_dim, or 2, 4, 6, 8; native K; default None.
        existing_hosts: 20 or 1000; native N; default 1000.
        reset: bool; clear the concrete buffer on restart; default True.
        Native message loss is 0.1; reward=time adds unit time cost until l=4."""
        if probes is not None and (
            type(probes) is not int or probes not in (2, 4, 6, 8)
        ):
            raise ValueError("probes must be None, 2, 4, 6 or 8")
        if type(existing_hosts) is not int or existing_hosts not in (20, 1000):
            raise ValueError("existing_hosts must be 20 or 1000")
        if type(reset) is not bool:
            raise ValueError("reset must be a bool")
        configure_prism(
            self,
            "zeroconf",
            state_dim,
            {"probes": probes, "existing_hosts": existing_hosts, "reset": reset},
            reward,
        )

    def _build_model(self):
        build_prism(self)
