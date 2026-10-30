"""Original QVBS csma JANI instance with native discounted rewards.

States are reachable valuations in Python BFS order; actions are local choices.
Unused action slots repeat the first choice. Reach targets absorb at zero reward.
Costs (Emin) are negated; rewards (Emax) retain their native scale.
state_dim selects the nearest published (N,K) variant by state-count ratio.
Targets outside the bundled range use its smallest or largest variant.
Finite property horizons are replaced by discounting, not reproduced.
Source authors, conversion and CC-BY 4.0 attribution: _qvbs/_data/model_manifest.json.
"""

from mdpforge.models._jani.model import JANIModel

METADATA = {
    "category": "service_operations",
    "description": (
        "QVBS JANI import in Python: CSMA/CD protocol; native transmission time cost."
    ),
    "reference": (
        "http://www.cs.bilkent.edu.tr/~tugrul/CS518/Papers/802.3- "
        "2000.pdf; QVBS (Hartmanns et al., TACAS 2019), CC-BY 4.0: "
        "https://qcomp.org/benchmarks/"
    ),
    "sizes": {
        "small": {
            "requested_state_dim": 1038,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 7958,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 66718,
            "parameters": {},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 1038}


class Model(JANIModel):
    """Construct the scalable csma source instance in Python."""

    benchmark = "csma"

    def __init__(self, state_dim: int = 1038, *, objective: str = "time_min"):
        """objective: time_max, time_min; selects the native reward property."""
        super().__init__(state_dim, objective=objective)
