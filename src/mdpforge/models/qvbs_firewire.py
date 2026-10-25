"""Original QVBS firewire JANI instance with native discounted rewards.

States are Storm's reachable valuations; actions are local choice offsets.
Unused action slots repeat the first choice. Reach targets absorb at zero reward.
Costs (Emin) are negated; rewards (Emax) retain their native scale.
state_dim scales delay>=1 using a power fit to published state counts.
Interpolation and extrapolation estimate sizes; Storm measures actual dimensions.
Finite property horizons are replaced by discounting, not reproduced.
Source authors, conversion and CC-BY 4.0 attribution: _qvbs_data/manifest.json.
"""

import stormpy  # noqa: F401

from mdpforge.models._qvbs import QVBSModel

METADATA = {
    "category": "service_operations",
    "description": (
        "QVBS JANI import via Storm: IEEE 1394 root contention; native "
        "election time cost."
    ),
    "reference": (
        "https://doi.org/10.1007/3-540-48778-6_4; QVBS (Hartmanns et "
        "al., TACAS 2019), CC-BY 4.0: https://qcomp.org/benchmarks/"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 4093,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 50000,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 212268,
            "parameters": {},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 4093}


class Model(QVBSModel):
    """Import the scalable firewire source instance using Storm."""

    benchmark = "firewire"

    def __init__(self, state_dim: int = 4093, *, objective: str = "time_min"):
        """objective: time_max, time_min, time_sending; selects the native reward
        property."""
        super().__init__(state_dim, objective=objective)
