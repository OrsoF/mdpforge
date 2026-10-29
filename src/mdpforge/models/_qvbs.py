"""Import pinned, native-reward QVBS MDPs without changing solver backends."""

import hashlib
import json
import math
from importlib.resources import files

from mdpforge.core.mdp import MDP
from mdpforge.models._jani import JaniProgram


def _select_configuration(source, requested_state_dim):
    """Select native parameters or the closest published size on a ratio scale."""
    if type(requested_state_dim) is not int or requested_state_dim < 1:
        raise ValueError("state_dim must be a positive integer")
    scaling = source["scaling"]
    if scaling["kind"] == "variants":
        variant = min(
            source["variants"],
            key=lambda item: (
                abs(math.log(item["states"]) - math.log(requested_state_dim)),
                item["states"],
            ),
        )
        return variant, dict(variant["constants"])

    constants = dict(source["constants"])
    if scaling["kind"] == "affine":
        slope, intercept = scaling["slope"], scaling["intercept"]
        value = (requested_state_dim - intercept + slope // 2) // slope
    elif scaling["kind"] == "power":
        anchors = scaling["anchors"]
        base_parameter, base_states = anchors[0]
        if len(anchors) == 1:
            value = (
                base_parameter * requested_state_dim + base_states // 2
            ) // base_states
        else:
            parameter, states = anchors[-1]
            exponent = math.log(states / base_states) / math.log(
                parameter / base_parameter
            )
            value = round(
                math.exp(
                    math.log(base_parameter)
                    + (math.log(requested_state_dim) - math.log(base_states)) / exponent
                )
            )
    elif scaling["kind"] == "quotas":
        base = scaling["states_per_counter_pair"]
        level = math.isqrt(requested_state_dim // base)
        value = min(
            {max(0, level - 1), level},
            key=lambda quota: (
                abs(base * (quota + 1) ** 2 - requested_state_dim),
                quota,
            ),
        )
    else:
        raise ValueError(f"Unknown QVBS scaling rule {scaling['kind']!r}")
    value = max(scaling["minimum"], value)
    if "maximum" in scaling:
        value = min(scaling["maximum"], value)
    for parameter in scaling["parameters"]:
        constants[parameter] = value
    return source, constants


class QVBSModel(MDP):
    """Use reachable state indices and state-local choice offsets."""

    def __init__(self, state_dim: int, *, objective: str):
        """Select a native configuration, then explore its actual dimensions."""
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        data = files("mdpforge.models").joinpath("_qvbs_data")
        manifest = json.loads(
            data.joinpath("manifest.json").read_text(encoding="utf-8")
        )
        source = manifest["models"][self.benchmark]
        if not isinstance(objective, str) or objective not in source["objectives"]:
            raise ValueError(f"objective must be one of {list(source['objectives'])}")
        variant, self.constants = _select_configuration(source, state_dim)
        self.source_file = variant["file"]
        raw = data.joinpath(self.source_file).read_bytes()
        self.source_sha256 = hashlib.sha256(raw).hexdigest()
        if self.source_sha256 != variant["sha256"]:
            raise ValueError(f"QVBS source checksum mismatch: {self.source_file}")
        document = json.loads(raw.decode("utf-8-sig"))
        if document["type"] != "mdp":
            raise ValueError("Only discrete-time, fully observable MDPs are supported")
        self.objective = objective
        self.source_revision = manifest["revision"]
        self._reward_spec = source["objectives"][objective]
        self.reward_name = self._reward_spec["exp"]
        self.reward_sign = -1 if self._reward_spec["op"] == "Emin" else 1
        for constant in document.get("constants", []):
            if constant["name"] in self.constants:
                constant["value"] = self.constants[constant["name"]]

        terminal = self._reward_spec.get("reach")
        if self._reward_spec.get("accumulate") not in (["steps"], ["exit"]):
            raise ValueError("Only native step or state-exit rewards are supported")
        compiled = JaniProgram(document, reward=self.reward_name, terminal=terminal)
        super().__init__(compiled.state_dim, compiled.action_dim)
        self.requested_state_dim = state_dim
        self.initial_states = compiled.initial_states
        self.terminal_states = compiled.terminal_states
        self._compiled = compiled
        self._set_name()

    def _set_name(self):
        """Separate cached matrices and values by construction provenance."""
        self.display_name = f"qvbs_{self.benchmark.replace('-', '_')}"
        config = self.get_config()
        config.pop("requested_state_dim")
        identity = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        self.name = (
            f"{self.state_dim}_{self.action_dim}_qvbs_{self.benchmark}"
            f"_v2_{self.objective}_{identity}"
        )

    def get_config(self):
        """Identify the source, constants, objective and construction version."""
        return {
            "benchmark": self.benchmark,
            "requested_state_dim": self.requested_state_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "source_revision": self.source_revision,
            "constants": self.constants,
            "objective": self.objective,
            "reward_spec": self._reward_spec,
            "reward_name": self.reward_name,
            "reward_sign": self.reward_sign,
            "adaptation_version": 2,
            "construction_backend": "python-jani",
            "construction_version": 2,
        }

    def create_model(self, check_transition=False, *, cache="readwrite"):
        """Finalize matrices and release source graphs, including after cache hits."""
        super().create_model(check_transition, cache=cache)
        if self._is_model_built():
            self._compiled = None

    def _build_model(self):
        """Preserve native expected rewards and duplicate unavailable action slots."""
        self.transition_matrix, self.reward_matrix = self._compiled.matrices(
            self.reward_sign
        )
