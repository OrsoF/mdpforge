"""Original QVBS consensus JANI instance with native discounted rewards.

States are Storm's reachable valuations; actions are local choice offsets.
Unused action slots repeat the first choice. Reach targets absorb at zero reward.
Costs (Emin) are negated; rewards (Emax) retain their native scale.
state_dim selects K>=1 for the two-process shared counter.
Finite property horizons are replaced by discounting, not reproduced.
Source authors, conversion and CC-BY 4.0 attribution: _qvbs_data/manifest.json.
"""

import stormpy  # noqa: F401

from mdpforge.models._qvbs import QVBSModel

METADATA = {
    "category": "abstract",
    "description": (
        "QVBS JANI import via Storm: Shared-coin consensus protocol; native step cost."
    ),
    "reference": (
        "https://doi.org/10.1016/0196-6774(90)90021-6; QVBS (Hartmanns "
        "et al., TACAS 2019), CC-BY 4.0: https://qcomp.org/benchmarks/"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 272,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 22656,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 100000,
            "parameters": {},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 272}


class Model(QVBSModel):
    """Import the scalable consensus source instance using Storm."""

    benchmark = "consensus"

    def __init__(self, state_dim: int = 272, *, objective: str = "steps_min"):
        """objective: steps_max, steps_min; selects the native reward property."""
        super().__init__(state_dim, objective=objective)
