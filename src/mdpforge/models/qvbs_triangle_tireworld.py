"""Triangle Tireworld from pinned JANI sources.

Reachable valuations and locations define states; enabled choices define actions.
Missing action slots repeat the first choice; stopped states absorb at zero.
Probability objectives pay a signed unit bonus on first reaching the target;
discounting replaces unbounded probabilities and outer verification assertions.
Native rewards retain their scale; minimisation objectives are negated.
Sources and parameters: _data/jani/qvbs/model_manifest.json.
"""

from mdpforge.models._jani_model import JANIModel

METADATA = {
    "category": "navigation",
    "description": "Triangle Tireworld",
    "reference": "http://users.cecs.anu.edu.au/~iain/icaps07.pdf",
    "model_type": "mdp",
    "source_collection": "qvbs",
    "source_url": "https://github.com/ahartmanns/qcomp/tree/a96cf379885c0a160a552a8fe0de417e2b55b161/benchmarks/mdp/triangle-tireworld",
    "variants": 5,
    "sizes": {
        "small": {
            "requested_state_dim": 80,
            "parameters": {

            },
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

TEST_PARAMETERS = {"state_dim": 80}


class Model(JANIModel):
    """Construct the bundled triangle-tireworld model with an explicit discounted objective."""

    collection = "qvbs"
    benchmark = "triangle-tireworld"

    def __init__(self, state_dim: int = 80, *, objective: str | None = None,
                 variant: str | None = None, constants: dict | None = None,
                 state_limit: int | None = 1_000_000):
        """objective: None or a source property; variant: None or a bundled basename.
        constants: None or source constant overrides; state_limit: positive int or None.
        state_dim: positive target; selects a published configuration when available."""
        super().__init__(state_dim, objective=objective, variant=variant,
                         constants=constants, state_limit=state_limit)
