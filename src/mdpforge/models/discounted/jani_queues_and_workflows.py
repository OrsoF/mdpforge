"""Finite two-server workflow reconstructed from the bundled Petri net.

States are (remaining_batches, queued_jobs, server0_phase, server1_phase).
An arrival adds two jobs; each server has two exponential service phases.
Immediate dispatches have priority and become discrete zero-reward actions;
timed races are uniformized and each completed job pays +1.
Each immediate action also consumes one discount step in this discrete task.
"""

import math
import xml.etree.ElementTree as ET

from mdpforge.models.discounted._jani.uniformization import (
    UniformizedModel,
    load_timed_source,
    positive_integer,
)

METADATA = {
    "category": "service_operations",
    "description": "Two-server finite workflow with immediate dispatch choices, uniformized service phases and completion rewards.",
    "reference": "Pinned JANI QueuesAndWorkflows/SimpleWorkflow Petri net and source export.",
    "model_type": "mdp",
    "source_model_type": "ma",
    "source_collection": "jani-models",
    "source_url": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/QueuesAndWorkflows",
    "sizes": {
        "tiny": {"requested_state_dim": 961, "parameters": {"batches": 10}},
        "small": {"requested_state_dim": 9409, "parameters": {"batches": 32}},
        "medium": {"requested_state_dim": 90601, "parameters": {"batches": 100}},
        "large": {"requested_state_dim": 361201, "parameters": {"batches": 200}},
    },
}

TEST_PARAMETERS = {"state_dim": 16, "batches": 1}


class Model(UniformizedModel):
    """Keep urgent dispatch decisions separate from uniformized timed events."""

    adaptation = {
        "method": "Petri-net dispatch actions with maximum progress and uniformized timed races",
        "source_correction": "PNPRO input arcs require >=1 token; exported JANI uses >1 and disables binary places",
        "objective": "+1 per completed job, discounted per discrete action or timed tick",
        "immediate_steps": "urgent and zero reward; each receives the same external per-step discount",
        "scaling": "finite initial batch count varies; two jobs per batch and source phase rates are retained",
    }

    def __init__(self, state_dim: int = 961, *, batches: int | None = None):
        """state_dim: positive target; batches: positive int or None, inferred."""
        positive_integer(state_dim, "state_dim")
        self.batches = positive_integer(
            max(1, (math.isqrt(state_dim) - 1) // 3) if batches is None else batches,
            "batches",
        )
        folder = "QueuesAndWorkflows/SimpleWorkflow/"
        _, export = load_timed_source(folder + "SimpleWorkflow.jani")
        raw, petri = load_timed_source(folder + "SimpleWorkflow.PNPRO")
        nodes = {
            node.attrib["name"]: node for node in ET.fromstring(raw).find("gspn/nodes")
        }
        self.arrival_rate = float(nodes["T2"].get("delay", 1))
        self.service_rates = tuple(
            float(nodes[name].get("delay", 1)) for name in ("T0", "T1", "T3", "T4")
        )
        rate = (
            self.arrival_rate
            + max(self.service_rates[:2])
            + max(self.service_rates[2:])
        )
        super().__init__(
            state_dim,
            (3 * self.batches + 1) ** 2,
            2,
            (self.batches, 0, 0, 0),
            rate,
            {
                "batches": self.batches,
                "jobs_per_batch": 2,
                "arrival_rate": self.arrival_rate,
                "service_phase_rates": self.service_rates,
            },
            [export, petri],
        )

    def _states(self):
        """Enumerate states consistent with the finite total number of jobs."""
        for incoming in range(self.batches + 1):
            arrived = 2 * (self.batches - incoming)
            for first in range(3):
                for second in range(3):
                    busy = int(first > 0) + int(second > 0)
                    for queued in range(arrived - busy + 1):
                        yield incoming, queued, first, second

    def _terminal(self, state):
        """Stop after all finite batches have been processed."""
        return state == (0, 0, 0, 0)

    def _rate_choices(self, state):
        """Dispatch urgently or race arrivals and the current service phases."""
        incoming, queued, first, second = state
        dispatches = []
        if queued:
            for server, phase in enumerate((first, second), 2):
                if phase == 0:
                    successor = list(state)
                    successor[1] -= 1
                    successor[server] = 1
                    dispatches.append(
                        [(tuple(successor), self.uniformization_rate, 0.0)]
                    )
        if dispatches:
            yield from dispatches
            return
        events = []
        if incoming:
            events.append(
                ((incoming - 1, queued + 2, first, second), self.arrival_rate, 0.0)
            )
        for server, phase in enumerate((first, second)):
            if phase:
                successor = list(state)
                successor[server + 2] = 2 if phase == 1 else 0
                events.append(
                    (
                        tuple(successor),
                        self.service_rates[2 * server + phase - 1],
                        float(phase == 2),
                    )
                )
        yield events
