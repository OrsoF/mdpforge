# Source: Hyon, E. and Jean-Marie, A. (2012). Scheduling services in a queuing system with impatience and setup costs. The Computer Journal 55(5):553-563.

import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import binom, poisson
from tqdm import trange

from core.model import GenericModel


def probability_surv(current_number, surviving_people, proba_to_stay):
    return binom.pmf(surviving_people, current_number, proba_to_stay)


def probability_arrival(
    arrivals: int,
    current_number: int,
    arrival_poisson_rate: float,
    max_queue_length: int,
):
    assert 0 <= arrivals <= max_queue_length - current_number

    if arrivals < max_queue_length - current_number:
        return poisson.pmf(arrivals, arrival_poisson_rate)
    else:
        return 1 - sum(
            poisson.pmf(arrivals, arrival_poisson_rate)
            for arrivals in range(max_queue_length - current_number)
        )


def probability_trajectory(
    current_number,
    surviving_people,
    arrival_number,
    proba_to_stay,
    arrival_poisson_rate,
    max_queue_length,
):
    if abs(current_number - surviving_people) >= 0.2 * max(100, current_number):
        return 0.0
    elif arrival_number >= 0.1 * (max_queue_length - current_number):
        return 0.0
    else:
        return probability_surv(
            current_number, surviving_people, proba_to_stay
        ) * probability_arrival(
            arrival_number, surviving_people, arrival_poisson_rate, max_queue_length
        )


def transition_function(
    yy: int,
    qq: int,
    zz: int,
    lambda_factor: float,
    max_queue_length: int,
    proba_to_stay: float,
):
    if abs(max(0, yy - qq - 1) - zz) >= 0.08 * max_queue_length:
        return 0.0
    else:
        current_number_of_person = max(0, yy - qq)
        return sum(
            probability_trajectory(
                current_number_of_person,
                surviving_people,
                zz - surviving_people,
                proba_to_stay,
                lambda_factor,
                max_queue_length,
            )
            for surviving_people in range(min(zz, current_number_of_person) + 1)
        )


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        state_dim = int(state_dim)
        self.proba_person_stay = 0.9
        self.arrival_poisson_rate = 0.9
        self.batch_cost = 1.5
        self.loss_cost = 2.0
        self.holding_cost = 0.1
        self.nb_serv = action_dim - 1  # action space dimension
        self.max_queue_length = state_dim - 1  # state space dimension

        self.q_cost = self.proba_person_stay * self.loss_cost + self.holding_cost

        self.state_dim = (
            self.max_queue_length + 1
        )  # On peut avoir 0 personnes ou max personnes dans la queue
        self.action_dim = (
            self.nb_serv + 1
        )  # On peut accueillir 0 personnes ou nb_serv personnes
        self.name = "{}_{}_queue_impatience".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))
        self.transition_matrix: list = [
            lil_matrix((self.state_dim, self.state_dim), dtype=float)
            for _ in range(self.action_dim)
        ]

        for yy in trange(self.state_dim):
            for qq in range(self.action_dim):
                self.reward_matrix[yy, qq] = qq * self.batch_cost + self.q_cost * max(
                    0, yy - qq
                )

                for zz in range(self.state_dim):
                    self.transition_matrix[qq][yy, zz] = transition_function(
                        yy,
                        qq,
                        zz,
                        self.arrival_poisson_rate,
                        self.max_queue_length,
                        self.proba_person_stay,
                    )
                row_sum = self.transition_matrix[qq][yy, :].sum()
                self.transition_matrix[qq][yy, yy] += 1.0 - row_sum

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
