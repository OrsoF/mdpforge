"""Ground small object instances of the two bundled PPDDL planning domains."""

import hashlib
import json
import re
from fractions import Fraction
from importlib.resources import files
from itertools import product


def _parse(text):
    stack, root = [], None
    for token in re.findall(r"[()]|[^\s()]+", re.sub(r";[^\n]*", "", text).lower()):
        if token == "(":
            node = []
            if stack:
                stack[-1].append(node)
            else:
                root = node
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("Unbalanced bundled PPDDL source")
            stack.pop()
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError("Unexpected bundled PPDDL token")
    if stack or not root or root[0] != "define":
        raise ValueError("Invalid bundled PPDDL source")
    return root


def _section(source, name):
    return next(
        item[1:]
        for item in source
        if isinstance(item, list) and item[0] == name
    )


def _typed(items):
    result, pending = [], []
    index = 0
    while index < len(items):
        if items[index] == "-":
            result.extend((name, items[index + 1]) for name in pending)
            pending = []
            index += 2
        else:
            pending.append(items[index])
            index += 1
    return result + [(name, "object") for name in pending]


def _and(expressions):
    result = True
    for expression in expressions:
        if expression is False:
            return False
        if expression is not True:
            result = (
                expression
                if result is True
                else {"op": "∧", "left": result, "right": expression}
            )
    return result


class _Grounder:
    def __init__(self, domain, problem, objects):
        self.objects = objects
        self.universe = set(objects["object"])
        self.actions = [
            item
            for item in domain
            if isinstance(item, list) and item[0] == ":action"
        ]
        self.initial = {
            tuple(atom)
            for atom in _section(problem, ":init")
            if atom[0] != "=" and set(atom[1:]) <= self.universe
        }
        self.dynamic = set()

        def modified(effect):
            op = effect[0]
            if op == "and":
                for child in effect[1:]:
                    modified(child)
            elif op == "probabilistic":
                for child in effect[2::2]:
                    modified(child)
            elif op == "forall":
                modified(effect[2])
            elif op == "not":
                self.dynamic.add(effect[1][0])
            elif op == "increase":
                if effect[1:] != [["total-cost"], "1"]:
                    raise ValueError("Unsupported PPDDL cost effect")
            else:
                self.dynamic.add(op)

        for action in self.actions:
            modified(action[action.index(":effect") + 1])
        self.variables = {}
        for predicate in _section(domain, ":predicates"):
            name = predicate[0]
            if name in self.dynamic:
                domains = [objects[kind] for _, kind in _typed(predicate[1:])]
                for arguments in product(*domains):
                    atom = (name, *arguments)
                    self.variables[atom] = "__".join(atom)

    def _atom(self, expression, binding):
        return (expression[0], *(binding.get(arg, arg) for arg in expression[1:]))

    def condition(self, expression, binding):
        op = expression[0]
        if op == "and":
            return _and(self.condition(child, binding) for child in expression[1:])
        if op == "not":
            operand = self.condition(expression[1], binding)
            return not operand if type(operand) is bool else {"op": "¬", "exp": operand}
        atom = self._atom(expression, binding)
        if not set(atom[1:]) <= self.universe:
            return False
        if op not in self.dynamic:
            return atom in self.initial
        return self.variables[atom]

    def effects(self, expression, binding):
        op = expression[0]
        if op == "and" or op == "forall":
            if op == "forall":
                typed = _typed(expression[1])
                children = [
                    (expression[2], binding | dict(zip((n for n, _ in typed), args)))
                    for args in product(*(self.objects[k] for _, k in typed))
                ]
            else:
                children = [(child, binding) for child in expression[1:]]
            branches = [(Fraction(1), frozenset(), frozenset())]
            for child, scope in children:
                branches = [
                    (p * q, add | other_add, delete | other_delete)
                    for p, add, delete in branches
                    for q, other_add, other_delete in self.effects(child, scope)
                ]
            return branches
        if op == "probabilistic":
            result, total = [], Fraction(0)
            for value, child in zip(expression[1::2], expression[2::2]):
                probability = Fraction(value)
                if not 0 <= probability <= 1:
                    raise ValueError("Invalid PPDDL probability")
                total += probability
                result.extend(
                    (probability * p, add, delete)
                    for p, add, delete in self.effects(child, binding)
                )
            if total > 1:
                raise ValueError("PPDDL probabilities exceed one")
            if total < 1:
                result.append((1 - total, frozenset(), frozenset()))
            return result
        if op == "increase":
            return [(Fraction(1), frozenset(), frozenset())]
        atom = self._atom(expression[1] if op == "not" else expression, binding)
        affected = frozenset({atom}) if set(atom[1:]) <= self.universe else frozenset()
        return [
            (
                Fraction(1),
                frozenset() if op == "not" else affected,
                affected if op == "not" else frozenset(),
            )
        ]

    def document(self, problem):
        edges = []
        for action in self.actions:
            options = dict(zip(action[2::2], action[3::2]))
            parameters = _typed(options.get(":parameters", []))
            for args in product(*(self.objects[kind] for _, kind in parameters)):
                binding = dict(zip((name for name, _ in parameters), args))
                guard = self.condition(options[":precondition"], binding)
                if guard is False:
                    continue
                destinations = []
                branches = self.effects(options[":effect"], binding)
                for probability, additions, deletions in branches:
                    if not probability:
                        continue
                    destinations.append(
                        {
                            "location": "ready",
                            "probability": {
                                "exp": {
                                    "op": "/",
                                    "left": probability.numerator,
                                    "right": probability.denominator,
                                }
                            },
                            "assignments": [
                                {"ref": self.variables[atom], "value": atom in additions}
                                for atom in sorted(additions | deletions)
                            ],
                        }
                    )
                edges.append(
                    {
                        "location": "ready",
                        "guard": {"exp": guard},
                        "destinations": destinations,
                    }
                )
        goals = _section(problem, ":goal")[0]
        if goals[0] != "and":
            goals = ["and", goals]
        retained = [atom for atom in goals[1:] if set(atom[1:]) <= self.universe]
        if not retained:
            raise ValueError("The chosen objects retain no original planning goal")
        goal = _and(self.condition(atom, {}) for atom in retained)
        document = {
            "jani-version": 1,
            "name": "scaled_ppddl",
            "type": "mdp",
            "features": ["derived-operators"],
            "variables": [
                {"name": name, "type": "bool", "initial-value": atom in self.initial}
                for atom, name in self.variables.items()
            ],
            "automata": [
                {
                    "name": "planning",
                    "locations": [{"name": "ready"}],
                    "initial-locations": ["ready"],
                    "edges": edges,
                }
            ],
            "system": {"elements": [{"automaton": "planning"}]},
        }
        return document, goal


def _count(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return value


def build_instance(family, parameters):
    """Ground original action schemas over a selected subset of source objects."""
    directory = files("mdpforge.models._qvbs").joinpath(
        "_data", "benchmarks", "mdp", family
    )
    prefix = "p01-b10-c5-dc0-fc0-dr0-gr1" if family == "boxworld" else "p01"
    raw_domain = directory.joinpath(prefix + "-domain.pddl").read_bytes()
    raw_problem = directory.joinpath(prefix + "-problem.pddl").read_bytes()
    domain, problem = _parse(raw_domain.decode()), _parse(raw_problem.decode())
    if family == "boxworld":
        limits = {
            "box": _count(parameters["boxes"], "boxes", 1, 10),
            "truck": _count(parameters["trucks"], "trucks", 1, 4),
            "plane": _count(parameters["planes"], "planes", 0, 2),
        }
        objects = {}
        for name, kind in _typed(_section(problem, ":objects")):
            group = objects.setdefault(kind, [])
            if kind not in limits or len(group) < limits[kind]:
                group.append(name)
        objects["object"] = [name for group in objects.values() for name in group]
    elif family == "random-predicates":
        count = _count(parameters["objects"], "objects", 1, 10)
        constants = _section(domain, ":constants")
        # OZ has both an initial self-predicate and an original unary goal.
        ordered = ["oz", *(name for name in constants if name != "oz")]
        objects = {"object": ordered[:count]}
    else:
        raise ValueError(f"Unsupported scalable PPDDL family {family!r}")
    document, goal = _Grounder(domain, problem, objects).document(problem)
    provenance = {
        "parameters": parameters,
        "objects": objects,
        "generator_version": 1,
        "domain_file": prefix + "-domain.pddl",
        "problem_file": prefix + "-problem.pddl",
        "domain_sha256": hashlib.sha256(raw_domain).hexdigest(),
        "problem_sha256": hashlib.sha256(raw_problem).hexdigest(),
        "generated_sha256": hashlib.sha256(
            json.dumps(document, sort_keys=True).encode()
        ).hexdigest(),
    }
    return document, goal, provenance
