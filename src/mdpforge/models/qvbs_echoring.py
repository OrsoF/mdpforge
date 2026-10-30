"""EchoRing from pinned JANI sources.

Reachable valuations and locations define states; enabled choices define actions.
Missing action slots repeat the first choice; stopped states absorb at zero.
Probability objectives pay a signed unit bonus on first reaching the target;
discounting replaces unbounded probabilities and outer verification assertions.
Native rewards retain their scale; minimisation objectives are negated.
Sources and parameters: _qvbs/_data/model_manifest.json.
"""

from mdpforge.models._jani.model import JANIModel

METADATA = {
    "category": "service_operations",
    "description": "EchoRing",
    "reference": "https://doi.org/10.1109/SRDS.2016.048",
    "model_type": "mdp",
    "source_collection": "qvbs",
    "source_url": "https://github.com/ahartmanns/qcomp/tree/a96cf379885c0a160a552a8fe0de417e2b55b161/benchmarks/mdp/echoring",
    "variants": 1,
    "sizes": {
        "small": {
            "requested_state_dim": 109515,
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

OBJECTIVES = ('MaxOffline1', 'MaxOffline2', 'MaxOffline3', 'MinFailed', 'MinOffline1', 'MinOffline2', 'MinOffline3')
VARIANTS = ('echoring.jani',)

TEST_PARAMETERS = {'state_dim': 109515, 'constants': {'ITERATIONS': 1}}


class Model(JANIModel):
    """Construct the bundled echoring model with an explicit discounted objective."""

    collection = "qvbs"
    benchmark = "echoring"

    def __init__(
        self,
        state_dim: int = 109515,
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
