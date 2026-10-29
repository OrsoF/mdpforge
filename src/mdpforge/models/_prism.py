"""Construct the bundled PRISM MDP commands using native CSP synchronisation.

Only the integer-variable command subset used by these pinned sources is read.
Reachable valuations are numbered in breadth-first order, starting at state 0.
Actions are state-local choice offsets; unused slots duplicate the first choice.
Native costs are negated and stopped at the source target predicate.
Optional first-passage adaptations absorb targets and add a unit step cost.
Discounting replaces source verification queries; no scheduler fairness is imposed.
Source attribution, licenses and counts are in _data/prism/manifest.json.
"""

import ast
import hashlib
import json
import keyword
import math
import re
from collections import defaultdict
from importlib.resources import files
from itertools import product

import numpy as np
from scipy.sparse import coo_matrix

from mdpforge.core.mdp import MDP


def _split(text, separator):
    parts, start, depth = [], 0, 0
    for index, character in enumerate(text):
        depth += (character == "(") - (character == ")")
        if character == separator and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _python_expression(text):
    conditional = _split(text, "?")
    if len(conditional) == 2:
        condition, branches = conditional
        yes, no = _split(branches, ":")
        text = f"({yes} if {condition} else {no})"
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"!(?!=)", " not ", text)
    text = re.sub(r"(?<![<>=!])=(?!=)", "==", text)
    text = text.replace("&", " and ").replace("|", " or ")
    return re.sub(
        r"\b[A-Za-z_]\w*\b",
        lambda match: (
            "_" + match[0]
            if keyword.iskeyword(match[0])
            and match[0] not in {"True", "False", "and", "or", "not", "if", "else"}
            else match[0]
        ),
        text,
    ).strip()


class _Names(ast.NodeTransformer):
    def __init__(self, constants, variables, formulas):
        self.constants, self.variables, self.formulas = constants, variables, formulas

    def visit_Name(self, node):
        if node.id in self.constants:
            return ast.Constant(self.constants[node.id])
        if node.id in self.variables:
            return ast.Subscript(
                ast.Name("v", ast.Load()),
                ast.Constant(self.variables[node.id]),
                ast.Load(),
            )
        if node.id in self.formulas:
            return self.visit(
                ast.parse(_python_expression(self.formulas[node.id]), mode="eval").body
            )
        if node.id not in {"min", "max", "floor", "pow"}:
            raise ValueError(f"Unsupported PRISM identifier {node.id!r}")
        return node


class _Program:
    def __init__(
        self,
        source,
        constants,
        reward,
        target,
        *,
        expand_formulas=False,
        step_cost=False,
    ):
        source = re.sub(r"//[^\n]*", "", source)
        if not re.match(r"\s*mdp\b", source) or re.search(r"\bsystem\b", source):
            raise ValueError("Only the bundled default-composition MDPs are supported")
        self.constants, self.variables, self.formulas = {}, {}, {}
        self.bounds, self.initial = [], []
        self.scope = {
            "__builtins__": {},
            "min": min,
            "max": max,
            "floor": math.floor,
            "pow": pow,
        }
        for name, expression in re.findall(
            r"const\s+(?:int|bool|double)\s+(\w+)\s*(?:=\s*([^;]+))?;", source
        ):
            if name in constants:
                value = constants[name]
            elif expression:
                value = self.expression(expression)(())
            else:
                raise ValueError(f"Missing PRISM constant {name}")
            self.constants[_python_expression(name)] = value
        self.formulas = dict(re.findall(r"formula\s+(\w+)\s*=\s*([^;]+);", source))
        self.formulas.update(re.findall(r'label\s+"([^"\n]+)"\s*=\s*([^;]+);', source))

        modules, remainder, cursor = {}, [], 0
        for match in re.finditer(r"\bmodule\s+(\w+)(.*?)\bendmodule\b", source, re.S):
            name, body = match.groups()
            remainder.append(source[cursor : match.start()])
            cursor = match.end()
            renamed = re.fullmatch(r"\s*=\s*(\w+)\s*\[(.*?)\]\s*", body, re.S)
            if renamed:
                original, mapping = renamed.groups()
                replacements = dict(
                    item.strip().split("=") for item in mapping.split(",")
                )
                replacements = {
                    key.strip(): value.strip() for key, value in replacements.items()
                }
                body = re.sub(
                    r"\b\w+\b",
                    lambda m: replacements.get(m[0], m[0]),
                    modules[original],
                )
            elif expand_formulas:
                for _ in range(len(self.formulas) + 1):
                    expanded = re.sub(
                        r"\b\w+\b",
                        lambda m: (
                            f"({self.formulas[m[0]]})"
                            if m[0] in self.formulas
                            else m[0]
                        ),
                        body,
                    )
                    if expanded == body:
                        break
                    body = expanded
                else:
                    raise ValueError("Cyclic PRISM formulas in module commands")
            modules[name] = body
        remainder.append(source[cursor:])
        self._declare("\n".join(remainder))
        for body in modules.values():
            self._declare(body)

        self.modules = []
        for body in modules.values():
            commands = defaultdict(list)
            for action, guard, updates in re.findall(
                r"\[(\w*)\]\s*([^;]*?)\s*->\s*([^;]+);", body, re.S
            ):
                branches = []
                for branch in _split(updates, "+"):
                    pieces = _split(branch, ":")
                    probability, update = pieces if len(pieces) == 2 else ("1", branch)
                    assignments = []
                    if update.strip() != "true":
                        for assignment in _split(update, "&"):
                            match = re.fullmatch(
                                r"\s*\(\s*(\w+)'\s*=\s*(.*)\)\s*", assignment, re.S
                            )
                            if match is None:
                                raise ValueError(
                                    f"Unsupported PRISM update {assignment!r}"
                                )
                            variable, expression = match.groups()
                            assignments.append(
                                (self.variables[variable], self.expression(expression))
                            )
                    branches.append((self.expression(probability), assignments))
                commands[action].append((self.expression(guard), branches))
            self.modules.append(commands)
        self.labels = dict.fromkeys(
            action for module in self.modules for action in module if action
        )
        self.participants = {
            action: [module[action] for module in self.modules if action in module]
            for action in self.labels
        }
        self.target = self.expression(target)
        structures = dict(
            re.findall(r'rewards\s+"([^"\n]+)"(.*?)endrewards', source, re.S)
        )
        if reward == "time" and not structures:
            structures["time"] = "[time] true : 1;"
        if step_cost:
            if reward in structures:
                raise ValueError("Adapted step cost would overwrite a native reward")
            structures[reward] = "true : 1;"
        if reward not in structures:
            raise ValueError(f"Unknown native reward {reward!r}")
        self.rewards = []
        for item in structures[reward].split(";"):
            if not item.strip():
                continue
            match = re.fullmatch(
                r"\s*(?:\[(\w*)\])?\s*(.*?)\s*:\s*(.*?)\s*", item, re.S
            )
            if match is None:
                raise ValueError(f"Unsupported PRISM reward {item!r}")
            action, guard, value = match.groups()
            self.rewards.append(
                (action, self.expression(guard), self.expression(value))
            )

    def expression(self, text):
        tree = ast.parse(_python_expression(text), mode="eval")
        tree = _Names(self.constants, self.variables, self.formulas).visit(tree)
        allowed = (
            ast.Expression,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.Subscript,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.UnaryOp,
            ast.Not,
            ast.USub,
            ast.UAdd,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Call,
            ast.IfExp,
        )
        if any(not isinstance(node, allowed) for node in ast.walk(tree)):
            raise ValueError(f"Unsupported PRISM expression {text!r}")
        code = compile(ast.fix_missing_locations(tree), "<bundled PRISM>", "eval")
        return lambda v: eval(code, self.scope, {"v": v})

    def _declare(self, body):
        for name, lower, upper, initial in re.findall(
            r"\b(\w+)\s*:\s*\[(.*?)\.\.(.*?)\]\s*(?:init\s+([^;]+))?;", body
        ):
            low, high = int(self.expression(lower)(())), int(self.expression(upper)(()))
            self.variables[name] = len(self.variables)
            self.bounds.append((low, high))
            self.initial.append(int(self.expression(initial)(())) if initial else low)

    def _distribution(self, state, commands):
        branches = [command[1] for command in commands]
        outcomes = defaultdict(float)
        for combination in product(*branches):
            probability = math.prod(branch[0](state) for branch in combination)
            if probability == 0:
                continue
            if not math.isfinite(probability) or probability < 0:
                raise ValueError("Invalid native transition probability")
            successor = list(state)
            written = set()
            for _, updates in combination:
                for variable, expression in updates:
                    value = expression(state)
                    low, high = self.bounds[variable]
                    if (
                        variable in written
                        or int(value) != value
                        or not low <= value <= high
                    ):
                        raise ValueError("Invalid simultaneous PRISM update")
                    written.add(variable)
                    successor[variable] = int(value)
            outcomes[tuple(successor)] += probability
        if not math.isclose(sum(outcomes.values()), 1, abs_tol=1e-12):
            raise ValueError("Native choice probabilities do not sum to one")
        return outcomes

    def choices(self, state):
        choices = []
        for module in self.modules:
            for command in module.get("", []):
                if command[0](state):
                    choices.append(("", self._distribution(state, [command])))
        for action, participants in self.participants.items():
            enabled = [
                [command for command in commands if command[0](state)]
                for commands in participants
            ]
            if all(enabled):
                choices.extend(
                    (action, self._distribution(state, commands))
                    for commands in product(*enabled)
                )
        if not choices:
            choices.append(("", {state: 1.0}))
        terminal = self.target(state)
        return [
            (
                outcomes,
                0.0
                if terminal
                else -sum(
                    value(state)
                    for label, guard, value in self.rewards
                    if (label is None or label == action) and guard(state)
                ),
            )
            for action, outcomes in choices
        ]


def configure_prism(model, family, requested, parameters, reward):
    """Select a published configuration without constructing its state graph."""
    if type(requested) is not int or requested < 1:
        raise ValueError("state_dim must be a positive integer")
    data = files("mdpforge.models").joinpath("_data", "prism")
    manifest = json.loads(data.joinpath("manifest.json").read_text(encoding="utf-8"))
    definition = manifest["models"][family]
    if reward not in definition["rewards"]:
        raise ValueError(f"reward must be one of {definition['rewards']}")
    candidates = [
        config
        for config in definition["configurations"]
        if all(
            value is None or config["parameters"].get(key) == value
            for key, value in parameters.items()
        )
    ]
    if not candidates:
        raise ValueError(f"Unsupported published {family} parameters: {parameters}")
    selected = min(candidates, key=lambda config: abs(config["states"] - requested))
    MDP.__init__(model, selected["states"], selected["actions"])
    model.requested_state_dim = requested
    for name, value in selected["parameters"].items():
        setattr(model, name, value)
    model.reward = reward
    model.initial_states = (0,)
    model.source_revision = definition.get("revision", manifest["revision"])
    if "sha256" in selected:
        model.source_sha256 = selected["sha256"]
        model.source_file = selected["file"]
        model.source_repository = definition["repository"]
    model._prism_configuration = {**definition.get("construction", {}), **selected}
    model._prism_target = definition["target"]
    suffix = "_".join(
        str(int(value)) if isinstance(value, bool) else str(value)
        for value in selected["parameters"].values()
    )
    model.name = f"{model.state_dim}_{model.action_dim}_{family}_2_{suffix}_{reward}"


def build_prism(model):
    """Explore the source commands and construct padded CSR choice matrices."""
    config = model._prism_configuration
    source = files("mdpforge.models").joinpath("_data", "prism", config["file"])
    raw = source.read_bytes()
    if "sha256" in config and hashlib.sha256(raw).hexdigest() != config["sha256"]:
        raise ValueError(f"PRISM source checksum mismatch: {config['file']}")
    text = raw.decode("utf-8")
    if config.get("legacy_mdp"):
        text = "mdp\n" + text
    program = _Program(
        text,
        config["constants"],
        model.reward,
        model._prism_target,
        expand_formulas=config.get("expand_formulas", False),
        step_cost=config.get("step_cost", False),
    )
    initial = tuple(program.initial)
    states, indices = [initial], {initial: 0}
    rows = [[] for _ in range(model.action_dim)]
    columns = [[] for _ in range(model.action_dim)]
    probabilities = [[] for _ in range(model.action_dim)]
    rewards = np.zeros((model.state_dim, model.action_dim))
    maximum = 0
    native_choices = native_transitions = 0
    terminal_states = (
        np.zeros(model.state_dim, dtype=bool) if config.get("absorb_target") else None
    )
    state_index = 0
    while state_index < len(states):
        choices = program.choices(states[state_index])
        terminal = terminal_states is not None and program.target(states[state_index])
        if terminal_states is not None:
            terminal_states[state_index] = terminal
        maximum = max(maximum, len(choices))
        native_choices += len(choices)
        native_transitions += sum(len(outcomes) for outcomes, _ in choices)
        if len(choices) > model.action_dim:
            raise ValueError("Native choices exceed the declared action dimension")
        for action in range(model.action_dim):
            outcomes, reward = choices[action] if action < len(choices) else choices[0]
            rewards[state_index, action] = reward
            for successor, probability in outcomes.items():
                target = indices.get(successor)
                if target is None:
                    target = len(states)
                    if target >= model.state_dim:
                        raise ValueError(
                            "Native states exceed the published state count"
                        )
                    indices[successor] = target
                    states.append(successor)
                rows[action].append(state_index)
                columns[action].append(state_index if terminal else target)
                probabilities[action].append(probability)
        state_index += 1
    if len(states) != model.state_dim:
        raise ValueError(f"Expected {model.state_dim} native states, got {len(states)}")
    model.transition_matrix = [
        coo_matrix(
            (probabilities[a], (rows[a], columns[a])),
            shape=(model.state_dim, model.state_dim),
        ).tocsr()
        for a in range(model.action_dim)
    ]
    model.reward_matrix = rewards
    model.native_max_choices = maximum
    model.native_choice_count = native_choices
    model.native_transition_count = native_transitions
    if terminal_states is not None:
        model.terminal_states = terminal_states
    return tuple(program.variables), states
