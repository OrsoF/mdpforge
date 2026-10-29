"""Published PRISM Rabin MDP with discounted cost to a designated process's entry.

States retain the original reachable integer valuations and atomic draw phases.
Actions are native local choices; unused action slots repeat the first choice.
The source variant omits the commented scheduling self-loops; no fairness is added.
The added reward is -1 per step until p[target_process]=2, then zero with absorption.
Native states are enumerated before absorption, preserving published dimensions.
state_dim selects the closest bundled three- or four-process configuration.
Original authors, GPL-2.0 source license and adaptation: _data/prism/manifest.json.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._prism import build_prism, configure_prism

METADATA = {
    "category": "abstract",
    "description": (
        "PRISM Rabin mutual exclusion: discounted steps until a designated "
        "process first enters its critical section."
    ),
    "reference": (
        "Rabin (1982), randomized mutual exclusion; PRISM case study: "
        "https://www.prismmodelchecker.org/casestudies/rabin.php; "
        "source examples: prismmodelchecker/prism, GPL-2.0."
    ),
    "sizes": {
        "small": {"requested_state_dim": 27766, "parameters": {"processes": 3}},
        "medium": {"requested_state_dim": 167209, "parameters": {"processes": 4}},
        "large": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": (
                "The next published case has 27,381,358 states; it is not bundled "
                "for explicit CSR construction."
            ),
        },
    },
}

TEST_PARAMETERS = {"state_dim": 27766, "processes": 3}


class Model(MDP):
    """Construct native Rabin transitions with an explicit first-passage objective."""

    def __init__(
        self,
        state_dim: int = 27766,
        *,
        processes: int | None = None,
        target_process: int = 1,
    ):
        """processes: None selects by state_dim, or 3/4; target_process: 1..processes."""
        if processes is not None and (
            type(processes) is not int or processes not in (3, 4)
        ):
            raise ValueError("processes must be None, 3 or 4")
        if type(target_process) is not int or target_process < 1:
            raise ValueError("target_process must be a positive integer")
        configure_prism(
            self, "prism_rabin", state_dim, {"processes": processes}, "steps_to_target"
        )
        if target_process > self.processes:
            raise ValueError("target_process must not exceed processes")
        self.target_process = target_process
        self._prism_target = f"p{target_process}=2"
        self.name += f"_target_{target_process}"

    def _build_model(self):
        build_prism(self)
