"""IEEE 802.11 Wireless LAN (with timer) from pinned JANI sources.

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
    "description": "IEEE 802.11 Wireless LAN (with timer)",
    "reference": "https://doi.org/10.1007/3-540-45605-8_11",
    "model_type": "mdp",
    "source_collection": "qvbs",
    "source_url": "https://github.com/ahartmanns/qcomp/tree/a96cf379885c0a160a552a8fe0de417e2b55b161/benchmarks/mdp/wlan_dl",
    "variants": 7,
    "sizes": {
        "small": {
            "requested_state_dim": 189703,
            "parameters": {

            },
        },
        "medium": {
            "requested_state_dim": 450627,
            "parameters": {

            },
        },
        "large": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": "No additional distinct published tractable size; runtime calibration is pending.",
        },
    },
}

OBJECTIVES = ('deadline',)
VARIANTS = ('wlan_dl.0.jani', 'wlan_dl.1.jani', 'wlan_dl.2.jani', 'wlan_dl.3.jani', 'wlan_dl.4.jani', 'wlan_dl.5.jani', 'wlan_dl.6.jani')

TEST_PARAMETERS = {'state_dim': 189703, 'constants': {'deadline': 1}}


class Model(JANIModel):
    """Construct the bundled wlan_dl model with an explicit discounted objective."""

    collection = "qvbs"
    benchmark = "wlan_dl"

    def __init__(
        self,
        state_dim: int = 189703,
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
