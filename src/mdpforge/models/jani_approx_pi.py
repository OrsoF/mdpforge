"""Monte Carlo approximation of pi from pinned JANI sources.

Reachable valuations and locations define states; enabled choices define actions.
Missing action slots repeat the first choice; stopped states absorb at zero.
Probability objectives pay a signed unit bonus on first reaching the target;
discounting replaces unbounded probabilities and outer verification assertions.
Native rewards retain their scale; minimisation objectives are negated.
The default objective accumulates native hits without normalising them into pi.
Sources and parameters: _jani/_data/model_manifest.json.
"""

from mdpforge.models._jani.model import JANIModel

METADATA = {
    "category": "abstract",
    "description": "Monte Carlo approximation of pi",
    "reference": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/ApproxPi",
    "model_type": "mdp",
    "source_collection": "jani-models",
    "source_url": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/ApproxPi",
    "variants": 40,
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

OBJECTIVES = ('hits',)
VARIANTS = ('approx_pi_00100_010_full.jani', 'approx_pi_00100_010_quadrant.jani', 'approx_pi_00100_050_full.jani', 'approx_pi_00100_050_quadrant.jani', 'approx_pi_00100_100_full.jani', 'approx_pi_00100_100_quadrant.jani', 'approx_pi_00100_500_full.jani', 'approx_pi_00100_500_quadrant.jani', 'approx_pi_00500_010_full.jani', 'approx_pi_00500_010_quadrant.jani', 'approx_pi_00500_050_full.jani', 'approx_pi_00500_050_quadrant.jani', 'approx_pi_00500_100_full.jani', 'approx_pi_00500_100_quadrant.jani', 'approx_pi_00500_500_full.jani', 'approx_pi_00500_500_quadrant.jani', 'approx_pi_01000_010_full.jani', 'approx_pi_01000_010_quadrant.jani', 'approx_pi_01000_050_full.jani', 'approx_pi_01000_050_quadrant.jani', 'approx_pi_01000_100_full.jani', 'approx_pi_01000_100_quadrant.jani', 'approx_pi_01000_500_full.jani', 'approx_pi_01000_500_quadrant.jani', 'approx_pi_05000_010_full.jani', 'approx_pi_05000_010_quadrant.jani', 'approx_pi_05000_050_full.jani', 'approx_pi_05000_050_quadrant.jani', 'approx_pi_05000_100_full.jani', 'approx_pi_05000_100_quadrant.jani', 'approx_pi_05000_500_full.jani', 'approx_pi_05000_500_quadrant.jani', 'approx_pi_10000_010_full.jani', 'approx_pi_10000_010_quadrant.jani', 'approx_pi_10000_050_full.jani', 'approx_pi_10000_050_quadrant.jani', 'approx_pi_10000_100_full.jani', 'approx_pi_10000_100_quadrant.jani', 'approx_pi_10000_500_full.jani', 'approx_pi_10000_500_quadrant.jani')

TEST_PARAMETERS = {"state_dim": 1}


class Model(JANIModel):
    """Construct the bundled ApproxPi model with an explicit discounted objective."""

    collection = "library"
    benchmark = "ApproxPi"

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
