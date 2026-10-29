"""Native PRISM firewire command semantics with discounted completion costs.

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
        "PRISM FireWire root contention with both wires, node control states "
        "and digital clocks."
    ),
    "reference": (
        "Kwiatkowska, Norman and Parker, The PRISM Benchmark Suite, QEST "
        "2012; CC-BY 4.0: "
        "https://github.com/prismmodelchecker/prism-benchmarks"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 4093,
            "parameters": {"delay": 3},
        },
        "medium": {
            "requested_state_dim": 212268,
            "parameters": {"delay": 36},
        },
        "large": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": (
                "The original concrete FireWire benchmark provides only delay=3 and "
                "delay=36; no third configuration is invented."
            ),
        },
    },
}

TEST_PARAMETERS = {"state_dim": 4093, "delay": 3}


class Model(MDP):
    """Construct the published reachable firewire MDP in Python."""

    def __init__(
        self,
        state_dim: int = 4093,
        *,
        delay: int | None = None,
        reward: str = "time",
    ):
        """delay: None selects by state_dim, or 3, 36; wire delay in 10 ns units.
        reward: time or time_sending; native cost; default time."""
        if delay is not None and (type(delay) is not int or delay not in (3, 36)):
            raise ValueError("delay must be None, 3 or 36")
        configure_prism(
            self,
            "firewire",
            state_dim,
            {"delay": delay},
            reward,
        )

    def _build_model(self):
        build_prism(self)
