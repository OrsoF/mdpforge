"""Shared sparse construction for finite adaptations of pinned timed sources."""

import hashlib
import json
import math
from importlib.resources import files

import numpy as np
from scipy.sparse import coo_matrix

from mdpforge.core.mdp import MDP


def positive_integer(value, name, minimum=1):
    """Require an integer parameter within its domain."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def load_timed_source(path):
    """Read a bundled source and verify its pinned checksum."""
    data = files("mdpforge.models.discounted._jani").joinpath("_data")
    manifest = json.loads(data.joinpath("IMPORT_MANIFEST.json").read_text("utf-8"))
    record = next(item for item in manifest["files"] if item["path"] == path)
    raw = data.joinpath(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"Source checksum mismatch: {path}")
    source = json.loads(raw.decode("utf-8-sig")) if path.endswith(".jani") else raw
    return source, {"file": path, "sha256": digest, "revision": manifest["revision"]}


class UniformizedModel(MDP):
    """Build finite rate models with one discrete step per uniformized event."""

    def __init__(
        self, requested, states, actions, initial_state, rate, parameters, sources
    ):
        positive_integer(requested, "state_dim")
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("uniformization_rate must be finite and positive")
        super().__init__(states, actions)
        self.requested_state_dim = requested
        self.initial_state = initial_state
        self.initial_states = (0,)
        self.uniformization_rate = float(rate)
        self.parameters = parameters
        self.sources = sources
        identity = json.dumps(
            {"parameters": parameters, "sources": sources, "version": 1},
            sort_keys=True,
        ).encode()
        fingerprint = int(hashlib.sha256(identity).hexdigest(), 16)
        recipe = type(self).__module__.rsplit(".", 1)[-1].replace("_", "")
        self.name = f"{states}_{actions}_{recipe}_1_{fingerprint}"

    def get_config(self):
        """Record source identities, the discrete objective and the event clock."""
        return {
            "requested_state_dim": self.requested_state_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "source_collection": "jani-models",
            "sources": self.sources,
            "parameters": self.parameters,
            "uniformization_rate": self.uniformization_rate,
            "adaptation": self.adaptation,
            "adaptation_version": 1,
            "discount_convention": "discount per discrete step; source time bounds are not retained",
        }

    def _build_model(self):
        """Enumerate finite states, divide event rates and complete each row."""
        states = list(self._ordered_states())
        if len(states) != self.state_dim:
            raise ValueError("The finite state enumeration disagrees with state_dim")
        indices = {state: index for index, state in enumerate(states)}
        rows = [[] for _ in range(self.action_dim)]
        columns = [[] for _ in range(self.action_dim)]
        data = [[] for _ in range(self.action_dim)]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))
        terminal_states = []
        for index, state in enumerate(states):
            terminal = self._terminal(state)
            if terminal:
                terminal_states.append(index)
            choices = [] if terminal else list(self._rate_choices(state))
            if len(choices) > self.action_dim:
                raise ValueError("The finite model has too many actions")
            for action in range(self.action_dim):
                events = (
                    choices[action if action < len(choices) else 0] if choices else []
                )
                outgoing = 0.0
                reward = 0.0 if terminal else self._state_reward(state)
                for successor, rate, impulse in events:
                    if not math.isfinite(rate) or rate < 0:
                        raise ValueError("Event rates must be finite and nonnegative")
                    if rate == 0:
                        continue
                    probability = rate / self.uniformization_rate
                    rows[action].append(index)
                    columns[action].append(indices[successor])
                    data[action].append(probability)
                    outgoing += probability
                    reward += probability * impulse
                if outgoing > 1 + 1e-12:
                    raise ValueError("The uniformization rate is below an exit rate")
                rows[action].append(index)
                columns[action].append(index)
                data[action].append(max(0.0, 1.0 - outgoing))
                self.reward_matrix[index, action] = reward
        self._terminal_state_indices = np.asarray(terminal_states, dtype=int)
        self.transition_matrix = [
            coo_matrix(
                (data[action], (rows[action], columns[action])),
                shape=(self.state_dim, self.state_dim),
            ).tocsr()
            for action in range(self.action_dim)
        ]
        for matrix in self.transition_matrix:
            matrix.eliminate_zeros()

    def _ordered_states(self):
        """Put the initial state first and retain the remaining domain order."""
        yield self.initial_state
        yield from (state for state in self._states() if state != self.initial_state)

    @property
    def terminal_states(self):
        """Recover declared terminal indices even when matrices came from cache."""
        if not hasattr(self, "_terminal_state_indices"):
            self._terminal_state_indices = np.asarray(
                [
                    index
                    for index, state in enumerate(self._ordered_states())
                    if self._terminal(state)
                ],
                dtype=int,
            )
        return self._terminal_state_indices

    def _terminal(self, state):
        """Identify an absorbing state when the recipe defines a stopped task."""
        return False

    def _state_reward(self, state):
        """Supply a per-tick state reward in addition to event impulses."""
        return 0.0
