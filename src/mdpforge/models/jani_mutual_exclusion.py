"""Pnueli-Zuck mutual exclusion from pinned JANI sources.

Reachable valuations and locations define states; enabled choices define actions.
Missing action slots repeat the first choice; stopped states absorb at zero.
Probability objectives pay a signed unit bonus on first reaching the target;
discounting replaces unbounded probabilities and outer verification assertions.
Native rewards retain their scale; minimisation objectives are negated.
Sources and parameters: _jani/_data/model_manifest.json.
"""

from mdpforge.models._jani.model import JANIModel

METADATA = {
    "category": "abstract",
    "description": "Pnueli-Zuck mutual exclusion",
    "reference": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/MutualExclusion",
    "model_type": "mdp",
    "source_collection": "jani-models",
    "source_url": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/MutualExclusion",
    "variants": 3,
    "sizes": {
        "small": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": "No additional distinct published tractable size; runtime calibration is pending.",
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

OBJECTIVES = ('property_1', 'property_2', 'property_3')
VARIANTS = ('mutual3.jani', 'mutual4.jani', 'mutual5.jani')

TEST_PARAMETERS = {"state_dim": 1}


class Model(JANIModel):
    """Construct the bundled MutualExclusion model with an explicit discounted objective."""

    collection = "library"
    benchmark = "MutualExclusion"

    def __init__(
        self,
        state_dim: int = 1,
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
