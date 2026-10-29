"""Construct individually catalogued models from pinned JANI source data."""

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


class JANIModel(MDP):
    """Use reachable state indices and state-local choice offsets."""

    collection = "qvbs"

    def __init__(
        self, state_dim: int, *, objective: str | None = None,
        variant: str | None = None, constants: dict | None = None,
        state_limit: int | None = None,
    ):
        """Select a native configuration, then explore its actual dimensions."""
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        data = files("mdpforge.models").joinpath("_data", "jani", self.collection)
        manifest = json.loads(
            data.joinpath("model_manifest.json").read_text(encoding="utf-8")
        )
        source = manifest["models"][self.benchmark]
        if "configurations" in source:
            candidates = source["configurations"]
            if variant is not None:
                if not isinstance(variant, str):
                    raise ValueError("variant must be a bundled filename or None")
                candidates = [
                    c for c in candidates
                    if variant in {c["file"], c["file"].rsplit("/", 1)[-1]}
                ]
                if not candidates:
                    raise ValueError(f"Unknown bundled JANI variant {variant!r}")
            known = [c for c in candidates if c["states"] is not None]
            if known:
                selected = min(
                    known,
                    key=lambda c: (
                        abs(math.log(c["states"]) - math.log(state_dim)),
                        c["states"],
                    ),
                )
            else:
                selected = min(candidates, key=lambda c: (c["bytes"], c["file"]))
            self.constants = dict(selected["constants"])
            specs = selected["objectives"]
        else:
            if variant is not None:
                raise ValueError("This model selects its native variant using state_dim")
            selected, self.constants = _select_configuration(source, state_dim)
            specs = source["objectives"]
        objective = source["default_objective"] if objective is None else objective
        if not isinstance(objective, str) or objective not in specs:
            raise ValueError(f"objective must be one of {list(specs)}")
        if constants is not None:
            if not isinstance(constants, dict):
                raise ValueError("constants must be a dictionary or None")
            self.constants.update(constants)
        self.source_file = selected["file"]
        raw = data.joinpath(self.source_file).read_bytes()
        self.source_sha256 = hashlib.sha256(raw).hexdigest()
        if self.source_sha256 != selected["sha256"]:
            raise ValueError(f"JANI source checksum mismatch: {self.source_file}")
        document = json.loads(raw.decode("utf-8-sig"))
        if document["type"] not in {"mdp", "dtmc"}:
            raise ValueError("Only finite MDPs and one-action DTMCs are supported")
        self.objective = objective
        self.source_revision = manifest["revision"]
        self._reward_spec = specs[objective]
        self.reward_name = self._reward_spec.get("exp")
        self.reward_sign = -1 if self._reward_spec["op"] in {"Emin", "Pmin"} else 1
        declared = {constant["name"] for constant in document.get("constants", [])}
        if self.constants.keys() - declared:
            raise ValueError("Unknown JANI source constant override")
        for constant in document.get("constants", []):
            if constant["name"] in self.constants:
                constant["value"] = self.constants[constant["name"]]

        terminal = self._reward_spec.get("reach")
        mode = self._reward_spec.get("mode", "native")
        if mode == "native" and self._reward_spec.get("accumulate", ["steps"]) not in (
            ["steps"], ["exit"]
        ):
            raise ValueError("Only native step or state-exit rewards are supported")
        compiled = JaniProgram(
            document, reward=self.reward_name, terminal=terminal, reward_mode=mode,
            continuation=self._reward_spec.get("continuation", True),
            state_limit=state_limit,
        )
        super().__init__(compiled.state_dim, compiled.action_dim)
        self.requested_state_dim = state_dim
        self.initial_states = compiled.initial_states
        self.terminal_states = compiled.terminal_states
        self._compiled = compiled
        self._set_name()

    def _set_name(self):
        """Separate cached matrices and values by construction provenance."""
        self.display_name = type(self).__module__.rsplit(".", 1)[-1]
        config = self.get_config()
        config.pop("requested_state_dim")
        identity = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        self.name = (
            f"{self.state_dim}_{self.action_dim}_{self.display_name}"
            f"_v2_{self.objective}_{identity}"
        )

    def get_config(self):
        """Identify the source, constants, objective and construction version."""
        return {
            "benchmark": self.benchmark,
            "source_collection": self.collection,
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
            "construction_version": 3,
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
