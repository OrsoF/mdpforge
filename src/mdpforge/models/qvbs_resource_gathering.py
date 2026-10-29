"""Original QVBS resource-gathering JANI instance with native discounted rewards.

States are reachable valuations in Python BFS order; actions are local choices.
Unused action slots repeat the first choice. Reach targets absorb at zero reward.
Costs (Emin) are negated; rewards (Emax) retain their native scale.
state_dim scales the two native collection quotas together (quota>=0).
The size estimate is 94*(quota+1)^2; the 5x5 world and rewards are preserved.
Finite property horizons are replaced by discounting, not reproduced.
Source authors, conversion and CC-BY 4.0 attribution: _qvbs_data/manifest.json.
"""

from mdpforge.models._qvbs import QVBSModel

METADATA = {
    "category": "navigation",
    "description": (
        "QVBS JANI import in Python: Resource gathering with attacks; "
        "native collected-gold reward."
    ),
    "reference": (
        "https://doi.org/10.1145/1390156.1390162; QVBS (Hartmanns et "
        "al., TACAS 2019), CC-BY 4.0: https://qcomp.org/benchmarks/"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 24064,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 90334,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 958894,
            "parameters": {},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 24064}


class Model(QVBSModel):
    """Construct the scalable resource-gathering source instance in Python."""

    benchmark = "resource-gathering"

    def __init__(self, state_dim: int = 24064, *, objective: str = "expgold"):
        """objective: expgold; selects the native reward property."""
        super().__init__(state_dim, objective=objective)
