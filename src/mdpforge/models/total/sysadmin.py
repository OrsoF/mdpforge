"""Stop the catalogue's SysAdmin network when every machine is operational.

State bits encode machine availability (1 means up) on a ring; action i
reboots machine i, and every action is valid in every state. The requested
state_dim (default 64, positive Python integer) is rounded down to a power
of two, with at least two machines: S = 2**max(2, floor(log2(state_dim))).
There is one action per machine and no random seed or time limit.

Nonterminal transitions reuse the discounted model exactly. At most one
machine flips per step. The rebooted machine has up probability 0.95;
other machines have up probability 0.95 when up or 0.02 when down, plus
0.03 per operational neighbor. Nonpositive flip weights are omitted,
positive weights are scaled together if their sum exceeds 0.9, and the
remaining probability leaves the state unchanged.

The all-up state S-1 becomes absorbing instead of allowing failures.
Rewards replace the discounted uptime reward: a nonterminal action earns
1 if it has positive probability of entering the goal, and 0 otherwise;
the goal earns 0. This is an action reward, not an entry probability.
Under every action each down machine has positive repair probability.
Thus every policy has a uniformly positive chance of reaching the goal
within the number of machines, giving finite expected absorption time
and finite total reward. Cycles prevent the current acyclic certificate.

Family reference: Guestrin, Koller, Parr & Venkataraman (2003), Efficient
Solution Algorithms for Factored MDPs; dynamics follow the local catalogue
adaptation, with an added all-up stopping goal and replaced rewards.
"""

from mdpforge.models.discounted.sysadmin import Model as DiscountedSysAdmin

METADATA = {
    "sizes": {
        "small": {"requested_state_dim": 2050, "parameters": {}},
        "medium": {"requested_state_dim": 32800, "parameters": {}},
        "large": {"requested_state_dim": 131200, "parameters": {}},
    },
    "category": "asset_management",
    "description": (
        "Restore all machines in a ring network to an absorbing all-up goal, "
        "rewarding actions that can reach it."
    ),
    "reference": (
        "Family reference: Guestrin, Koller, Parr & Venkataraman (2003). "
        "Efficient Solution Algorithms for Factored MDPs."
    ),
}

TEST_PARAMETERS = {"state_dim": 8}


class Model(DiscountedSysAdmin):
    """SysAdmin with an absorbing all-up goal and finite total reward."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 64):
        """Accept state_dim as a positive Python integer target (default 64)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name += "_total_all_up_entry_reward_v1"
        self.terminal_states = (self.state_dim - 1,)

    def _build_model(self):
        """Reuse network dynamics, reward goal access and absorb the all-up state."""
        super()._build_model()
        goal = self.terminal_states[0]
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = transition[:, goal].toarray().ravel() > 0
            transition = transition.tolil()
            transition[goal, :] = 0.0
            transition[goal, goal] = 1.0
            self.transition_matrix[action] = transition
        self.reward_matrix[goal, :] = 0.0
