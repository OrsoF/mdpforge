# Source: Winston, W. L. and Goldberg, J. B. (2004). Operations Research: Applications and Algorithms, 4th ed.

import numpy as np
import scipy.stats as st
from scipy.sparse import lil_matrix

from core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self._stock_size = (state_dim - 1) // 2  # Size of the stock
        self.max_action = action_dim - 1  # Max Number of action - 1
        self.BinomialeP = 0.4  #
        self.BinomialeQ = 2  # Size of the demand
        self._holding_cost = 4  # holding cost
        self._shortage_cost = 2  # shortageCost
        self._action_cost = 5  # action cost

        self.name = "{}_{}_inventory".format(state_dim, action_dim)

    def _build_model(self):
        self.state_dim = 2 * self._stock_size + 1
        self.action_dim = self.max_action + 1

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.empty((self.state_dim, self.action_dim))

        self._build_transition_matrix()
        self._build_reward_matrix()

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _build_transition_matrix(self):
        sortie = 0
        # look over all the actions
        for aa in range(self.action_dim):
            for ss1 in range(self.state_dim):
                etat = ss1 - self._stock_size
                etat = min(
                    etat + aa, self._stock_size
                )  # on doit pas sortir apres une action
                proba = 0
                borne = min(self.BinomialeQ, etat + self._stock_size)
                for y in range(
                    borne + 1
                ):  # dans tous les cas on atteint pas la borne inferieure
                    proba = st.binom.pmf(y, self.BinomialeQ, self.BinomialeP)
                    sortie = etat - y  # valeur sortie
                    self.transition_matrix[aa][ss1, self._stock_size + sortie] = proba
                if (
                    borne < self.BinomialeQ
                ):  # on est dans un cas ou la borne inf est atteinte
                    proba = 0
                    sortie = -self._stock_size
                    for y in range(borne, self.BinomialeQ + 1):
                        proba += st.binom.pmf(y, self.BinomialeQ, self.BinomialeP)
                    self.transition_matrix[aa][ss1, self._stock_size + sortie] = proba

    def _build_reward_matrix(self):
        for ss1 in range(self.state_dim):
            for aa in range(self.action_dim):
                res = 0.0
                etat = ss1 - self._stock_size
                if etat < 0:
                    # case of shortage
                    res += -self._shortage_cost * etat + aa * self._action_cost
                else:
                    # case of holding
                    res += self._holding_cost * etat + aa * self._action_cost
                self.reward_matrix[ss1, aa] = res
