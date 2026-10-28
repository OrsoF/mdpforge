"""Original PRISM Lehmann--Rabin MDP with discounted cost to a designated eater.

States retain the original reachable integer valuations and all twelve phases.
Actions are native local choices; unused action slots repeat the first choice.
The original variant retains scheduling self-loops; no fairness is imposed.
The added reward is -1 per step until the designated p is 8 or 9 (native eat),
then zero with absorption; native states are enumerated before this adaptation.
state_dim selects a bundled three-, four- or five-philosopher configuration.
Original authors, GPL-2.0 source license and adaptation: _prism_data/manifest.json.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._prism import build_prism, configure_prism

METADATA = {
    "category": "abstract",
    "description": (
        "PRISM Lehmann-Rabin philosophers: discounted steps until a designated "
        "philosopher first reaches the native eating phases."
    ),
    "reference": (
        "Lehmann and Rabin (1981), On the Advantages of Free Choice; "
        "https://www.prismmodelchecker.org/casestudies/phil.php; "
        "source examples: prismmodelchecker/prism, GPL-2.0."
    ),
    "sizes": {
        "small": {"requested_state_dim": 956, "parameters": {"philosophers": 3}},
        "medium": {"requested_state_dim": 9440, "parameters": {"philosophers": 4}},
        "large": {"requested_state_dim": 93068, "parameters": {"philosophers": 5}},
    },
}

TEST_PARAMETERS = {"state_dim": 956, "philosophers": 3}


class Model(MDP):
    """Construct the original philosopher protocol with first-passage costs."""

    def __init__(
        self,
        state_dim: int = 956,
        *,
        philosophers: int | None = None,
        target_philosopher: int = 1,
    ):
        """philosophers: None selects by state_dim, or 3/4/5; target_philosopher: 1..N."""
        if philosophers is not None and (
            type(philosophers) is not int or philosophers not in (3, 4, 5)
        ):
            raise ValueError("philosophers must be None, 3, 4 or 5")
        if type(target_philosopher) is not int or target_philosopher < 1:
            raise ValueError("target_philosopher must be a positive integer")
        configure_prism(
            self,
            "prism_dining_philosophers",
            state_dim,
            {"philosophers": philosophers},
            "steps_to_target",
        )
        if target_philosopher > self.philosophers:
            raise ValueError("target_philosopher must not exceed philosophers")
        self.target_philosopher = target_philosopher
        self._prism_target = f"p{target_philosopher}>=8 & p{target_philosopher}<=9"
        self.name += f"_target_{target_philosopher}"

    def _build_model(self):
        build_prism(self)
