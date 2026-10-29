"""Bounded Exponential Backoff from pinned JANI sources.

Reachable valuations and locations define states; enabled choices define actions.
Missing action slots repeat the first choice; stopped states absorb at zero.
Probability objectives pay a signed unit bonus on first reaching the target;
discounting replaces unbounded probabilities and outer verification assertions.
Native rewards retain their scale; minimisation objectives are negated.
Sources and parameters: _data/jani/qvbs/model_manifest.json.
"""

from mdpforge.models._jani_model import JANIModel

METADATA = {
    "category": "service_operations",
    "description": "Bounded Exponential Backoff",
    "reference": "https://doi.org/10.1007/978-3-642-21461-5_4",
    "model_type": "mdp",
    "source_collection": "qvbs",
    "source_url": "https://github.com/ahartmanns/qcomp/tree/a96cf379885c0a160a552a8fe0de417e2b55b161/benchmarks/mdp/beb",
    "variants": 4,
    "sizes": {
        "small": {
            "requested_state_dim": 4528,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": "No additional distinct published tractable size; runtime calibration is pending.",
        },
        "large": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": "No additional distinct published tractable size; runtime calibration is pending.",
        },
    },
}

OBJECTIVES = ('GaveUp', 'LineSeized')
VARIANTS = ('beb.3-4.jani', 'beb.4-8.jani', 'beb.5-16.jani', 'beb.6-16.jani')

TEST_PARAMETERS = {'state_dim': 4528, 'constants': {'N': 1}}


class Model(JANIModel):
    """Construct the bundled beb model with an explicit discounted objective."""

    collection = "qvbs"
    benchmark = "beb"

    def __init__(
        self,
        state_dim: int = 4528,
        *,
        objective: str | None = None,
        variant: str | None = None,
        constants: dict | None = None,
        state_limit: int | None = 1_000_000,
    ):
        """objective: None or OBJECTIVES; variant: None or VARIANTS; selects source rules.
        constants: None or source constant overrides; state_limit: positive int or None.
        state_dim: positive target; selects a published configuration when available."""
        super().__init__(
            state_dim,
            objective=objective,
            variant=variant,
            constants=constants,
            state_limit=state_limit,
        )
