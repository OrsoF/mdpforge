"""Original QVBS eajs JANI instance with native discounted rewards.

States are Storm's reachable valuations; actions are local choice offsets.
Unused action slots repeat the first choice. Reach targets absorb at zero reward.
Costs (Emin) are negated; rewards (Emax) retain their native scale.
state_dim scales energy_capacity>=1 from the published size anchor.
This linear size estimate does not guarantee an exact reachable-state count.
Finite property horizons are replaced by discounting, not reproduced.
Source authors, conversion and CC-BY 4.0 attribution: _qvbs_data/manifest.json.
"""

import stormpy  # noqa: F401

from mdpforge.models._qvbs import QVBSModel

METADATA = {
    "category": "service_operations",
    "description": (
        "QVBS JANI import via Storm: Energy-aware job scheduling; "
        "native on-time task utility."
    ),
    "reference": (
        "https://doi.org/10.1007/978-3-319-06200-6_24; QVBS (Hartmanns "
        "et al., TACAS 2019), CC-BY 4.0: https://qcomp.org/benchmarks/"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 12828,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 50000,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 200000,
            "parameters": {},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 12828}


class Model(QVBSModel):
    """Import the scalable eajs source instance using Storm."""

    benchmark = "eajs"

    def __init__(self, state_dim: int = 12828, *, objective: str = "ExpUtil"):
        """objective: ExpUtil; selects the native reward property."""
        super().__init__(state_dim, objective=objective)
