"""Import pinned, native-reward QVBS MDPs without changing solver backends."""

import hashlib
import json
import math
from importlib.resources import files

import numpy as np
from scipy.sparse import coo_matrix

from mdpforge.core.mdp import MDP


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
    """Use reachable Storm state indices and state-local choice offsets."""

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
        import stormpy

        self.storm_version = stormpy.__version__
        self.source_revision = manifest["revision"]
        self._reward_spec = source["objectives"][objective]
        self.reward_name = self._reward_spec["exp"]
        self.reward_sign = -1 if self._reward_spec["op"] == "Emin" else 1
        for constant in document.get("constants", []):
            if constant["name"] in self.constants:
                constant["value"] = self.constants[constant["name"]]

        # Preserve only a Boolean stopping predicate; bounded source properties
        # are verification queries, not part of the native transition dynamics.
        terminal = self._reward_spec.get("reach")
        document["properties"] = (
            [
                {
                    "name": "mdpforge_terminal",
                    "expression": {
                        "op": "filter",
                        "fun": "values",
                        "states": {"op": "initial"},
                        "values": terminal,
                    },
                }
            ]
            if terminal is not None
            else []
        )
        program, properties = stormpy.parse_jani_model_from_string(
            json.dumps(document, ensure_ascii=False)
        )
        options = stormpy.BuilderOptions([p.raw_formula for p in properties])
        options.set_build_all_reward_models()
        options.set_build_all_labels()
        compiled = stormpy.build_sparse_model_with_options(program, options)
        if compiled.model_type != stormpy.ModelType.MDP:
            raise ValueError("Storm did not construct an MDP")
        if not compiled.has_reward_model(self.reward_name):
            raise ValueError(f"Missing native reward model {self.reward_name!r}")

        choices = np.asarray(compiled.nondeterministic_choice_indices, dtype=np.int64)
        super().__init__(int(compiled.nr_states), max(1, int(np.diff(choices).max())))
        self.requested_state_dim = state_dim
        self.initial_states = tuple(int(s) for s in compiled.initial_states)
        self.terminal_states = np.zeros(self.state_dim, dtype=bool)
        if properties:
            truth = stormpy.model_checking(compiled, properties[0]).get_truth_values()
            self.terminal_states[list(truth)] = True
        self._compiled = compiled
        config = self.get_config()
        config.pop("requested_state_dim")
        identity = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        self.name = (
            f"{self.state_dim}_{self.action_dim}_qvbs_{self.benchmark}"
            f"_v2_{objective}_{identity}"
        )

    def get_config(self):
        """Identify the source, constants, objective and Storm enumeration version."""
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
            "storm_version": self.storm_version,
            "adaptation_version": 2,
        }

    def create_model(self, check_transition=False, *, cache="readwrite"):
        """Finalize matrices and release Storm objects, including after cache hits."""
        super().create_model(check_transition, cache=cache)
        if self._is_model_built():
            self._compiled = None

    def _build_model(self):
        """Preserve native expected rewards and duplicate unavailable action slots."""
        compiled = self._compiled
        matrix = compiled.transition_matrix
        native = compiled.get_reward_model(self.reward_name)
        native.reduce_to_state_based_rewards(matrix, False)
        if native.has_transition_rewards:
            raise ValueError("Storm did not reduce transition rewards to expectations")
        state_rewards = (
            np.asarray(native.state_rewards, dtype=float)
            if native.has_state_rewards
            else np.zeros(self.state_dim)
        )
        choice_rewards = (
            np.asarray(native.state_action_rewards, dtype=float)
            if native.has_state_action_rewards
            else np.zeros(int(compiled.nr_choices))
        )
        choices = np.asarray(compiled.nondeterministic_choice_indices, dtype=np.int64)
        rows = [[] for _ in range(self.action_dim)]
        columns = [[] for _ in range(self.action_dim)]
        probabilities = [[] for _ in range(self.action_dim)]
        rewards = np.zeros((self.state_dim, self.action_dim))
        for state in range(self.state_dim):
            start, end = int(choices[state]), int(choices[state + 1])
            terminal = self.terminal_states[state] or start == end
            for action in range(self.action_dim):
                if terminal:
                    rows[action].append(state)
                    columns[action].append(state)
                    probabilities[action].append(1.0)
                    continue
                # Repeating an enabled choice adds no policy or value options;
                # a dummy zero-reward action would let a policy evade real costs.
                choice = start + action if action < end - start else start
                for entry in matrix.get_row(choice):
                    rows[action].append(state)
                    columns[action].append(int(entry.column))
                    probabilities[action].append(float(entry.value()))
                rewards[state, action] = self.reward_sign * (
                    state_rewards[state] + choice_rewards[choice]
                )
        self.transition_matrix = [
            coo_matrix(
                (probabilities[a], (rows[a], columns[a])),
                shape=(self.state_dim, self.state_dim),
            ).tocsr()
            for a in range(self.action_dim)
        ]
        self.reward_matrix = rewards
