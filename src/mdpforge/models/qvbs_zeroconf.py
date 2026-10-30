"""IPv4 Zeroconf Protocol from pinned JANI sources.

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
    "description": "IPv4 Zeroconf Protocol",
    "reference": "https://www.rfc-editor.org/info/rfc3927",
    "model_type": "mdp",
    "source_collection": "qvbs",
    "source_url": "https://github.com/ahartmanns/qcomp/tree/a96cf379885c0a160a552a8fe0de417e2b55b161/benchmarks/mdp/zeroconf",
    "variants": 1,
    "sizes": {
        "small": {
            "requested_state_dim": 670,
            "parameters": {

            },
        },
        "medium": {
            "requested_state_dim": 1924,
            "parameters": {

            },
        },
        "large": {
            "requested_state_dim": 796941,
            "parameters": {

            },
        },
    },
}

OBJECTIVES = ('correct_max', 'correct_min')
VARIANTS = ('zeroconf.jani',)

TEST_PARAMETERS = {'state_dim': 670, 'constants': {'N': 1, 'K': 1, 'reset': True}}


class Model(JANIModel):
    """Construct the bundled zeroconf model with an explicit discounted objective."""

    collection = "qvbs"
    benchmark = "zeroconf"

    def __init__(
        self,
        state_dim: int = 670,
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
