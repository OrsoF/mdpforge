"""Original QVBS firewire_abst JANI instance with native discounted rewards.

States are Storm's reachable valuations; actions are local choice offsets.
Unused action slots repeat the first choice. Reach targets absorb at zero reward.
Costs (Emin) are negated; rewards (Emax) retain their native scale.
state_dim selects delay in 1..168, bounded by the native clock.
The reachable size saturates at 1436 states; larger targets select that limit.
Finite property horizons are replaced by discounting, not reproduced.
Source authors, conversion and CC-BY 4.0 attribution: _qvbs_data/manifest.json.
"""

import stormpy  # noqa: F401

from mdpforge.models._qvbs import QVBSModel

METADATA = {
    "category": "service_operations",
    "description": (
        "QVBS JANI import via Storm: Abstract IEEE 1394 root "
        "contention; native round cost."
    ),
    "reference": (
        "https://doi.org/10.1007/3-540-48778-6_4; QVBS (Hartmanns et "
        "al., TACAS 2019), CC-BY 4.0: https://qcomp.org/benchmarks/"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 611,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 1024,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 1436,
            "parameters": {},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 611}


class Model(QVBSModel):
    """Import the scalable firewire_abst source instance using Storm."""

    benchmark = "firewire_abst"

    def __init__(self, state_dim: int = 611, *, objective: str = "rounds"):
        """objective: rounds, time_max, time_min; selects the native reward
        property."""
        super().__init__(state_dim, objective=objective)
