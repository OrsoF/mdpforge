"""Original QVBS wlan JANI instance with native discounted rewards.

States are reachable valuations in Python BFS order; actions are local choices.
Unused action slots repeat the first choice. Reach targets absorb at zero reward.
Costs (Emin) are negated; rewards (Emax) retain their native scale.
state_dim selects the nearest published backoff variant (0..6) by size ratio.
Targets outside the bundled range use its smallest or largest variant.
Finite property horizons are replaced by discounting, not reproduced.
Source authors, conversion and CC-BY 4.0 attribution: _qvbs_data/manifest.json.
"""

from mdpforge.models._qvbs import QVBSModel

METADATA = {
    "category": "service_operations",
    "description": (
        "QVBS JANI import in Python: IEEE 802.11 wireless LAN; native "
        "transmission cost."
    ),
    "reference": (
        "https://doi.org/10.1007/3-540-45605-8_11; QVBS (Hartmanns et "
        "al., TACAS 2019), CC-BY 4.0: https://qcomp.org/benchmarks/"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 2954,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 28480,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 345000,
            "parameters": {},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 2954}


class Model(QVBSModel):
    """Construct the scalable wlan source instance in Python."""

    benchmark = "wlan"

    def __init__(self, state_dim: int = 2954, *, objective: str = "cost_min"):
        """objective: cost_max, cost_min, num_collisions, time_max, time_min;
        selects the native reward property."""
        super().__init__(state_dim, objective=objective)
