"""Dining Philosophers from pinned JANI sources.

Reachable valuations and locations define states; enabled choices define actions.
Missing action slots repeat the first choice; stopped states absorb at zero.
Probability objectives pay a signed unit bonus on first reaching the target;
discounting replaces unbounded probabilities and outer verification assertions.
Native rewards retain their scale; minimisation objectives are negated.
Sources and parameters: _data/jani/qvbs/model_manifest.json.
"""

from mdpforge.models._jani_model import JANIModel

METADATA = {
    "category": "abstract",
    "description": "Dining Philosophers",
    "reference": "https://doi.org/10.1145/567532.567547",
    "model_type": "mdp",
    "source_collection": "qvbs",
    "source_url": "https://github.com/ahartmanns/qcomp/tree/a96cf379885c0a160a552a8fe0de417e2b55b161/benchmarks/mdp/philosophers-mdp",
    "variants": 4,
    "sizes": {
        "small": {
            "requested_state_dim": 440,
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

OBJECTIVES = ('eat',)
VARIANTS = ('philosophers-mdp.10.jani', 'philosophers-mdp.20.jani', 'philosophers-mdp.3.jani', 'philosophers-mdp.30.jani')

TEST_PARAMETERS = {"state_dim": 440}


class Model(JANIModel):
    """Construct the bundled philosophers-mdp model with an explicit discounted objective."""

    collection = "qvbs"
    benchmark = "philosophers-mdp"

    def __init__(
        self,
        state_dim: int = 440,
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
