"""Elevators from pinned JANI sources.

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
    "description": "Elevators",
    "reference": "http://www.aaai.org/Library/AIPS/2000/aips00-036.php",
    "model_type": "mdp",
    "source_collection": "qvbs",
    "source_url": "https://github.com/ahartmanns/qcomp/tree/a96cf379885c0a160a552a8fe0de417e2b55b161/benchmarks/mdp/elevators",
    "variants": 4,
    "sizes": {
        "small": {
            "requested_state_dim": 909,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 14796,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 539136,
            "parameters": {},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 909}


class Model(JANIModel):
    """Construct the bundled elevators model with an explicit discounted objective."""

    collection = "qvbs"
    benchmark = "elevators"

    def __init__(
        self,
        state_dim: int = 909,
        *,
        objective: str | None = None,
        variant: str | None = None,
        constants: dict | None = None,
        state_limit: int | None = 1_000_000,
    ):
        """objective: None or a source property; variant: None or a bundled basename.
        constants: None or source constant overrides; state_limit: positive int or None.
        state_dim: positive target; selects a published configuration when available."""
        super().__init__(
            state_dim,
            objective=objective,
            variant=variant,
            constants=constants,
            state_limit=state_limit,
        )
