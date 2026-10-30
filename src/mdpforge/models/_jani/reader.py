"""Explore the finite JANI automata networks bundled with the QVBS models."""

import math
import operator
from collections import defaultdict
from copy import deepcopy
from itertools import product

import numpy as np
from scipy.sparse import coo_matrix

_BINARY = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "=": operator.eq,
    "≠": operator.ne,
    "<": operator.lt,
    "≤": operator.le,
    ">": operator.gt,
    "≥": operator.ge,
    "min": min,
    "max": max,
}


class JANIStateLimitError(ValueError):
    """A source graph exceeds the explicitly configured construction limit."""


def _fields(document, allowed):
    """Reject unsupported fields instead of silently changing model semantics."""
    unknown = document.keys() - {*allowed, "comment"}
    if unknown:
        raise ValueError(f"Unsupported JANI fields: {sorted(unknown)}")


def _globalise(document):
    """Give local variables distinct global names while preserving their scope."""
    document = deepcopy(document)
    variables = document.setdefault("variables", [])
    occupied = {variable["name"] for variable in variables}
    for index, automaton in enumerate(document["automata"]):
        mapping = {
            variable["name"]: f"__local_{index}_{variable['name']}"
            for variable in automaton.get("variables", [])
        }
        if occupied.intersection(mapping.values()):
            raise ValueError("Local JANI variable name collision")
        occupied.update(mapping.values())

        def expression(value):
            if isinstance(value, str):
                return mapping.get(value, value)
            if isinstance(value, list):
                return [expression(item) for item in value]
            if isinstance(value, dict):
                return {
                    key: child
                    if key in {"op", "function", "comment"}
                    else expression(child)
                    for key, child in value.items()
                }
            return value

        def assignments(items):
            for item in items:
                item["ref"] = mapping.get(item["ref"], item["ref"])
                item["value"] = expression(item["value"])

        for variable in automaton.get("variables", []):
            variable["name"] = mapping[variable["name"]]
            variable["type"] = expression(variable["type"])
            if "initial-value" in variable:
                variable["initial-value"] = expression(variable["initial-value"])
            variables.append(variable)
        automaton["variables"] = []
        if "restrict-initial" in automaton:
            automaton["restrict-initial"] = expression(automaton["restrict-initial"])
        for location in automaton["locations"]:
            assignments(location.get("transient-values", []))
        for edge in automaton["edges"]:
            if "guard" in edge:
                edge["guard"] = expression(edge["guard"])
            for destination in edge["destinations"]:
                if "probability" in destination:
                    destination["probability"] = expression(destination["probability"])
                assignments(destination.get("assignments", []))
    return document


class JaniProgram:
    """Build reachable choices with synchronised updates and native rewards."""

    def __init__(
        self,
        document,
        *,
        reward=None,
        terminal=None,
        reward_mode="native",
        continuation=True,
        state_limit=None,
    ):
        document = _globalise(document)
        _fields(
            document,
            {
                "jani-version",
                "name",
                "metadata",
                "type",
                "features",
                "actions",
                "constants",
                "functions",
                "variables",
                "automata",
                "system",
                "properties",
                "restrict-initial",
            },
        )
        if document["jani-version"] != 1 or document["type"] not in {"mdp", "dtmc"}:
            raise ValueError("Only finite JANI version 1 MDPs and DTMCs are supported")
        if reward_mode not in {"native", "reachability", "step_cost", "state"}:
            raise ValueError("Unknown JANI reward adaptation")
        if state_limit is not None and (
            type(state_limit) is not int or state_limit < 1
        ):
            raise ValueError("state_limit must be a positive integer or None")
        self.reward_mode = reward_mode
        self.state_limit = state_limit
        if set(document.get("features", [])) - {
            "derived-operators",
            "functions",
            "state-exit-rewards",
        }:
            raise ValueError("Unsupported JANI features")
        restrictions = []
        for scope in (document, *document["automata"]):
            restriction = scope.get("restrict-initial", {"exp": True})
            _fields(restriction, {"exp"})
            restrictions.append(restriction["exp"])

        self.constants = {}
        self.variables = {}
        self.functions = {}
        self._integer_bounds = {}
        for constant in document.get("constants", []):
            _fields(constant, {"name", "type", "value"})
            name = constant["name"]
            if name in self.constants or "value" not in constant:
                raise ValueError(f"Duplicate or undefined JANI constant {name!r}")
            value = self._expression(constant["value"])(self.constants)
            self._check_value(name, constant["type"], value)
            self.constants[name] = value

        for variable in document.get("variables", []):
            _fields(variable, {"name", "type", "transient", "initial-value"})
            name = variable["name"]
            if name in self.variables or name in self.constants:
                raise ValueError(f"Duplicate JANI identifier {name!r}")
            kind = variable["type"]
            if not variable.get("transient", False) and not (
                kind == "bool"
                or (
                    isinstance(kind, dict)
                    and kind.get("kind") == "bounded"
                    and kind.get("base") == "int"
                )
            ):
                raise ValueError("Persistent JANI variables must be Boolean or bounded")
            self.variables[name] = variable
        for function in document.get("functions", []):
            _fields(function, {"name", "type", "parameters", "body"})
            name = function["name"]
            if (
                name in self.functions
                or name in self.variables
                or name in self.constants
            ):
                raise ValueError(f"Duplicate JANI identifier {name!r}")
            parameters = []
            for parameter in function["parameters"]:
                _fields(parameter, {"name", "type"})
                if parameter["name"] in parameters:
                    raise ValueError("Duplicate JANI function parameter")
                if parameter["type"] not in {"bool", "int", "real"}:
                    raise ValueError("Unsupported JANI function parameter type")
                parameters.append(parameter["name"])
            if function["type"] not in {"bool", "int", "real"}:
                raise ValueError("Unsupported JANI function return type")
            self.functions[name] = (parameters, function["body"])
        self.defaults, initial_domains = self._initial_values(restrictions)
        for name, value in self.defaults.items():
            self._check_value(name, self.variables[name]["type"], value)
        self.state_variables = tuple(
            name for name, v in self.variables.items() if not v.get("transient", False)
        )
        self.transients = self.variables.keys() - set(self.state_variables)
        if reward_mode == "native":
            if reward not in self.transients or type(self.defaults[reward]) is bool:
                raise ValueError("The reward must be a numeric transient JANI variable")
            if self.defaults[reward] != 0:
                raise ValueError("Only zero-default transient rewards are supported")
        self.reward = reward
        self.state_reward = self._expression(reward) if reward_mode == "state" else None
        self.terminal = self._expression(terminal if terminal is not None else False)
        self.continuation = self._expression(continuation)

        self.actions = set()
        for action in document.get("actions", []):
            _fields(action, {"name"})
            if action["name"] in self.actions:
                raise ValueError("Duplicate JANI action")
            self.actions.add(action["name"])
        system = document["system"]
        _fields(system, {"elements", "syncs"})
        definitions = {a["name"]: a for a in document["automata"]}
        if len(definitions) != len(document["automata"]):
            raise ValueError("Duplicate JANI automaton")
        automata = []
        names = set()
        for element in system["elements"]:
            _fields(element, {"automaton"})
            name = element["automaton"]
            if name not in definitions or name in names:
                raise ValueError("Unknown or repeated JANI system automaton")
            names.add(name)
            automata.append(definitions[name])
        if not automata or names != definitions.keys():
            raise ValueError("Each JANI automaton must occur once in the system")
        self.locations, self.edges = [], []
        initial_locations = []
        for automaton in automata:
            locations, edges = self._automaton(automaton)
            self.locations.append(locations)
            self.edges.append(edges)
            initial = automaton["initial-locations"]
            if not initial or len(set(initial)) != len(initial):
                raise ValueError("JANI initial locations must be nonempty and unique")
            if any(location not in locations for location in initial):
                raise ValueError("Unknown JANI initial location")
            initial_locations.append(initial)
        self.syncs = []
        for sync in system.get("syncs", []):
            _fields(sync, {"synchronise", "result"})
            vector = sync["synchronise"]
            if len(vector) != len(automata):
                raise ValueError("Unsupported JANI synchronisation vector")
            participants = tuple((i, a) for i, a in enumerate(vector) if a is not None)
            if not participants or any(a not in self.actions for _, a in participants):
                raise ValueError("Empty or undeclared JANI synchronisation action")
            if participants in self.syncs or sync.get("result") not in self.actions | {
                None
            }:
                raise ValueError("Duplicate or undeclared JANI synchronisation action")
            self.syncs.append(participants)
        predicates = [self._expression(exp) for exp in restrictions]
        self.states = []
        for locations in product(*initial_locations):
            for valuation in product(
                *(initial_domains[n] for n in self.state_variables)
            ):
                state = (locations, *valuation)
                values = self._environment(state)
                if all(predicate(values) for predicate in predicates):
                    self.states.append(state)
                    if state_limit is not None and len(self.states) > state_limit:
                        raise JANIStateLimitError(
                            f"JANI initial states exceed state_limit={state_limit}"
                        )
        if not self.states:
            raise ValueError("The JANI initial restrictions admit no initial state")
        self.initial_states = tuple(range(len(self.states)))
        self._explore()
        if document["type"] == "dtmc" and self.action_dim != 1:
            raise ValueError("DTMC source produced nondeterministic choices")

    def _initial_values(self, restrictions):
        """Resolve initial equalities and enumerate remaining finite initial domains."""
        known, pending = {}, []
        for name, variable in self.variables.items():
            if "initial-value" in variable:
                pending.append((name, variable["initial-value"]))

        def equalities(exp):
            if not isinstance(exp, dict):
                return
            if exp.get("op") == "∧":
                equalities(exp["left"])
                equalities(exp["right"])
            elif exp.get("op") == "=":
                for key, other in (("left", "right"), ("right", "left")):
                    name = exp[key]
                    if isinstance(name, str) and name in self.variables:
                        pending.append((name, exp[other]))
                        break

        for exp in restrictions:
            equalities(exp)
        while pending:
            deferred = []
            for name, exp in pending:
                if name in known:
                    continue
                try:
                    known[name] = self._expression(exp)(self.constants | known)
                except KeyError:
                    deferred.append((name, exp))
            if len(deferred) == len(pending):
                break
            pending = deferred
        domains = {}
        for name, variable in self.variables.items():
            if name in known:
                domains[name] = (known[name],)
            elif variable.get("transient", False):
                raise ValueError(f"Missing initial transient value for {name!r}")
            elif variable["type"] == "bool":
                domains[name] = (False, True)
                known[name] = False
            else:
                kind = variable["type"]
                lower = self._expression(kind["lower-bound"])(self.constants)
                upper = self._expression(kind["upper-bound"])(self.constants)
                domains[name] = range(lower, upper + 1)
                known[name] = lower
        return known, domains

    def _automaton(self, automaton):
        """Compile a bundled automaton with global variables and named actions."""
        _fields(
            automaton,
            {
                "name",
                "variables",
                "locations",
                "initial-locations",
                "edges",
                "restrict-initial",
            },
        )
        if automaton.get("variables"):
            raise ValueError("The bundled JANI subset uses only global variables")
        locations = {}
        for location in automaton["locations"]:
            _fields(location, {"name", "transient-values"})
            name = location["name"]
            if name in locations:
                raise ValueError("Duplicate JANI location")
            assignments = self._assignments(location.get("transient-values", []))
            if any(ref not in self.transients for ref, _, _ in assignments):
                raise ValueError("Location assignments must target transient variables")
            locations[name] = assignments
        edges = defaultdict(lambda: defaultdict(list))
        for edge in automaton["edges"]:
            _fields(edge, {"location", "action", "guard", "destinations"})
            if edge["location"] not in locations:
                raise ValueError("Unknown JANI source location")
            if "action" in edge and edge["action"] not in self.actions:
                raise ValueError("Unknown JANI edge action")
            guard = edge.get("guard", {"exp": True})
            _fields(guard, {"exp"})
            destinations = []
            for destination in edge["destinations"]:
                _fields(destination, {"location", "probability", "assignments"})
                if destination["location"] not in locations:
                    raise ValueError("Unknown JANI destination location")
                probability = destination.get("probability", {"exp": 1})
                _fields(probability, {"exp"})
                destinations.append(
                    (
                        destination["location"],
                        self._expression(probability["exp"]),
                        self._assignments(destination.get("assignments", [])),
                    )
                )
            if not destinations:
                raise ValueError("A JANI edge must have a destination")
            edges[edge["location"]][edge.get("action")].append(
                (self._expression(guard["exp"]), destinations)
            )
        return locations, edges

    def _expression(self, expression, bindings=None, stack=()):
        """Compile the supported pure expressions without evaluating source code."""
        if type(expression) in (bool, int, float):
            if not math.isfinite(expression):
                raise ValueError("Nonfinite JANI literal")
            return lambda values: expression
        if isinstance(expression, str):
            if bindings is not None and expression in bindings:
                return bindings[expression]
            if expression in self.constants:
                value = self.constants[expression]
                return lambda values: value
            if expression not in self.constants and expression not in self.variables:
                raise ValueError(f"Unknown JANI identifier {expression!r}")
            return lambda values: values[expression]
        if not isinstance(expression, dict):
            raise ValueError(f"Unsupported JANI expression {expression!r}")
        op = expression.get("op")
        if op == "call":
            _fields(expression, {"op", "function", "args"})
            name = expression["function"]
            if name not in self.functions or name in stack:
                raise ValueError(f"Unknown or recursive JANI function {name!r}")
            parameters, body = self.functions[name]
            if len(parameters) != len(expression["args"]):
                raise ValueError(f"Wrong argument count for JANI function {name!r}")
            arguments = {
                parameter: self._expression(argument, bindings, stack)
                for parameter, argument in zip(parameters, expression["args"])
            }
            return self._expression(body, arguments, (*stack, name))
        if op == "ite":
            _fields(expression, {"op", "if", "then", "else"})
            condition = self._expression(expression["if"], bindings, stack)
            yes = self._expression(expression["then"], bindings, stack)
            no = self._expression(expression["else"], bindings, stack)
            return lambda values: yes(values) if condition(values) else no(values)
        if op == "¬":
            _fields(expression, {"op", "exp"})
            operand = self._expression(expression["exp"], bindings, stack)
            return lambda values: not operand(values)
        if op not in _BINARY and op not in {"∧", "∨"}:
            raise ValueError(f"Unsupported JANI operator {op!r}")
        _fields(expression, {"op", "left", "right"})
        left = self._expression(expression["left"], bindings, stack)
        right = self._expression(expression["right"], bindings, stack)
        if op == "∧":
            return lambda values: left(values) and right(values)
        if op == "∨":
            return lambda values: left(values) or right(values)
        operation = _BINARY[op]
        return lambda values: operation(left(values), right(values))

    def _check_value(self, name, kind, value):
        """Enforce finite scalar types and declared integer bounds."""
        if isinstance(kind, dict):
            _fields(kind, {"kind", "base", "lower-bound", "upper-bound"})
            if kind["kind"] != "bounded" or kind["base"] != "int":
                raise ValueError(f"Unsupported JANI type for {name!r}")
            if name not in self._integer_bounds:
                lower = self._expression(kind["lower-bound"])(self.constants)
                upper = self._expression(kind["upper-bound"])(self.constants)
                if type(lower) is not int or type(upper) is not int or lower > upper:
                    raise ValueError(f"Invalid JANI integer bounds for {name!r}")
                self._integer_bounds[name] = lower, upper
            lower, upper = self._integer_bounds[name]
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"JANI value outside integer bounds for {name!r}")
        elif kind == "bool":
            if type(value) is not bool:
                raise ValueError(f"Expected Boolean JANI value for {name!r}")
        elif kind in {"int", "real"}:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Expected finite numeric JANI value for {name!r}")
            if kind == "int" and type(value) is not int:
                raise ValueError(f"Expected integer JANI value for {name!r}")
        else:
            raise ValueError(f"Unsupported JANI type for {name!r}")

    def _assignments(self, assignments):
        """Compile assignments that execute atomically at each ordered index."""
        result, seen = [], set()
        for assignment in assignments:
            _fields(assignment, {"ref", "value", "index"})
            name = assignment["ref"]
            index = assignment.get("index", 0)
            if not isinstance(name, str) or name not in self.variables:
                raise ValueError(f"Unsupported or duplicate JANI assignment {name!r}")
            if type(index) is not int or index < 0 or (name, index) in seen:
                raise ValueError("Invalid or duplicate indexed JANI assignment")
            seen.add((name, index))
            result.append((name, index, self._expression(assignment["value"])))
        return result

    def _environment(self, state):
        values = self.constants | self.defaults
        values.update(zip(self.state_variables, state[1:]))
        return values

    def _update(self, values, assignments):
        groups = defaultdict(list)
        for name, index, expression in assignments:
            groups[index].append((name, expression))
        for index in sorted(groups):
            updates = {}
            for name, expression in groups[index]:
                value = expression(values)
                self._check_value(name, self.variables[name]["type"], value)
                if name in updates and updates[name] != value:
                    raise ValueError(f"Conflicting JANI updates for {name!r}")
                updates[name] = value
            values = values | updates
        return values

    def _enabled_choices(self, locations, values):
        """Keep silent edges independent and combine matching synchronisation edges."""
        enabled = []
        for i, location in enumerate(locations):
            actions = {}
            for action, edges in self.edges[i][location].items():
                actions[action] = [d for guard, d in edges if guard(values)]
            enabled.append(actions)
            for destinations in actions.get(None, []):
                yield ((i, destinations),)
        for participants in self.syncs:
            alternatives = [enabled[i].get(action, []) for i, action in participants]
            for selection in product(*alternatives):
                yield tuple((i, d) for (i, _), d in zip(participants, selection))

    def _distribution(self, destinations, values):
        """Validate a choice and normalize source rounding within absolute 1e-6."""
        result, total = [], 0.0
        for location, probability, assignments in destinations:
            p = float(probability(values))
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Invalid JANI transition probability")
            total += p
            if p:
                result.append((location, p, assignments))
        if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"JANI choice probabilities sum to {total:.17g}; "
                    "expected 1 within absolute tolerance 1e-6"
                )
            result = [
                (location, probability / total, assignments)
                for location, probability, assignments in result
            ]
        return result

    def _explore(self):
        """Enumerate the native graph before applying mdpforge terminal absorption."""
        indices = {state: index for index, state in enumerate(self.states)}
        self.choices, terminals, failures = [], [], []
        index = 0
        while index < len(self.states):
            state = self.states[index]
            base = self._environment(state)
            assignments = [
                assignment
                for i, location in enumerate(state[0])
                for assignment in self.locations[i][location]
            ]
            values = self._update(base, assignments)
            terminals.append(bool(self.terminal(values)))
            failures.append(not self.continuation(values))
            choices = []
            for selected in self._enabled_choices(state[0], values):
                distributions = [self._distribution(d, values) for _, d in selected]
                # Add the source-state reward once and average destination rewards.
                outcomes = {}
                if self.reward_mode == "native":
                    expected_reward = float(values[self.reward])
                elif self.reward_mode == "state":
                    expected_reward = float(self.state_reward(values))
                else:
                    expected_reward = float(self.reward_mode == "step_cost")
                for branch in product(*distributions):
                    p = math.prod(destination[1] for destination in branch)
                    locations = list(state[0])
                    assignments = []
                    for (i, _), (location, _, updates) in zip(selected, branch):
                        locations[i] = location
                        assignments.extend(updates)
                    # Transients reset before updates and never enter the state key.
                    updated = self._update(base, assignments)
                    successor = (
                        tuple(locations),
                        *(updated[n] for n in self.state_variables),
                    )
                    target = indices.get(successor)
                    if target is None:
                        target = len(self.states)
                        if self.state_limit is not None and target >= self.state_limit:
                            raise JANIStateLimitError(
                                "JANI exploration exceeds "
                                f"state_limit={self.state_limit}; select a smaller "
                                "variant or raise the explicit construction limit"
                            )
                        indices[successor] = target
                        self.states.append(successor)
                    outcomes[target] = outcomes.get(target, 0.0) + p
                    if self.reward_mode == "native":
                        expected_reward += p * updated[self.reward]
                if not math.isfinite(expected_reward):
                    raise ValueError("Nonfinite JANI expected reward")
                choices.append((outcomes, expected_reward))
            self.choices.append(choices)
            index += 1
        self.state_dim = len(self.states)
        self.action_dim = max(1, max(map(len, self.choices)))
        self.terminal_states = np.asarray(terminals, dtype=bool)
        if self.reward_mode == "reachability":
            goals = self.terminal_states.copy()
            self.terminal_states |= np.asarray(failures, dtype=bool)
            for state, choices in enumerate(self.choices):
                self.choices[state] = [
                    (
                        outcomes,
                        sum(p for target, p in outcomes.items() if goals[target]),
                    )
                    for outcomes, _ in choices
                ]

    def matrices(self, reward_sign):
        """Construct padded choices with zero-reward absorbing terminal states."""
        rows = [[] for _ in range(self.action_dim)]
        columns = [[] for _ in range(self.action_dim)]
        probabilities = [[] for _ in range(self.action_dim)]
        rewards = np.zeros((self.state_dim, self.action_dim))
        for state, choices in enumerate(self.choices):
            for action in range(self.action_dim):
                if self.terminal_states[state] or not choices:
                    outcomes, reward = {state: 1.0}, 0.0
                else:
                    outcomes, reward = choices[action if action < len(choices) else 0]
                rewards[state, action] = reward_sign * reward
                for target, probability in outcomes.items():
                    rows[action].append(state)
                    columns[action].append(target)
                    probabilities[action].append(probability)
        matrices = [
            coo_matrix(
                (probabilities[a], (rows[a], columns[a])),
                shape=(self.state_dim, self.state_dim),
            )
            for a in range(self.action_dim)
        ]
        return matrices, rewards
