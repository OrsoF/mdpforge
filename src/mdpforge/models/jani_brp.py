"""Bounded retransmission protocol from pinned JANI sources.

Reachable valuations and locations define states; enabled choices define actions.
Missing action slots repeat the first choice; stopped states absorb at zero.
Probability objectives pay a signed unit bonus on first reaching the target;
discounting replaces unbounded probabilities and outer verification assertions.
Native rewards retain their scale; minimisation objectives are negated.
Sources and parameters: _data/jani/library/model_manifest.json.
"""

from mdpforge.models._jani_model import JANIModel

METADATA = {
    "category": "service_operations",
    "description": "Bounded retransmission protocol",
    "reference": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/BRP",
    "model_type": "dtmc",
    "source_collection": "jani-models",
    "source_url": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/BRP",
    "variants": 1,
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

TEST_PARAMETERS = {"state_dim": 1}


class Model(JANIModel):
    """Construct the bundled BRP model with an explicit discounted objective."""

    collection = "library"
    benchmark = "BRP"

    def __init__(self, state_dim: int = 1, *, objective: str | None = None,
                 variant: str | None = None, constants: dict | None = None,
                 state_limit: int | None = 1_000_000):
        """objective: None or a source property; variant: None or a bundled basename.
        constants: None or source constant overrides; state_limit: positive int or None.
        state_dim: positive target; selects a published configuration when available."""
        super().__init__(state_dim, objective=objective, variant=variant,
                         constants=constants, state_limit=state_limit)
