"""Original QVBS eajs JANI instance with native discounted rewards.

States are reachable valuations in Python BFS order; actions are local choices.
Unused action slots repeat the first choice. Reach targets absorb at zero reward.
Costs (Emin) are negated; rewards (Emax) retain their native scale.
state_dim scales energy_capacity>=1 from the published size anchor.
This linear size estimate does not guarantee an exact reachable-state count.
Finite property horizons are replaced by discounting, not reproduced.
Source authors, conversion and CC-BY 4.0 attribution: _data/jani/qvbs/model_manifest.json.
"""

from mdpforge.models._jani_model import JANIModel

METADATA = {
    "category": "service_operations",
    "description": (
        "QVBS JANI import in Python: Energy-aware job scheduling; "
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


class Model(JANIModel):
    """Construct the scalable eajs source instance in Python."""

    benchmark = "eajs"

    def __init__(self, state_dim: int = 12828, *, objective: str = "ExpUtil"):
        """objective: ExpUtil; selects the native reward property."""
        super().__init__(state_dim, objective=objective)
