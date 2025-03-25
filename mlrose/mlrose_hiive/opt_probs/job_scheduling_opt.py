""" Classes for defining optimization problem objects."""

# Author: Qingchuan Lyu (based on Genevieve Hayes and Andrew Rollings' mlrose code)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers import OnePointCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.fitness import JobScheduling
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt


class JobSchedulingOpt(DiscreteOpt):
    """
    :param schedule: A permutation of task indices that represents the order in which tasks are scheduled.
                    e.g., schedule = [2, 0, 1, 3, 4]
    :param tasks: an array. e.g.,
                    Task ID, Duration, Deadline, Profit
                    tasks = [
                        Task('A', 2, 5, 50),
                        Task('B', 3, 8, 70),
                        Task('C', 1, 4, 40),
                        Task('D', 2, 6, 60),
                        Task('E', 4, 10, 100)
                    ]
    :param break_time: an integer
    :param work_limit: an integer
    """
    def __init__(self, length=None, fitness_fn=None, maximize=True,
                 crossover=None, mutator=None, tasks=None, break_time=None, work_limit=None):

        if (fitness_fn is None) and (length is None):
            raise Exception("fitness_fn or length must be specified.")

        if length is None:
            length = len(fitness_fn.weights)

        self.length = length

        if (tasks is None) | (break_time is None) | (work_limit is None):
            raise Exception("""tasks, break_time and work_limit must exist.""")

        self.tasks = tasks
        self.break_time = break_time
        self.work_limit = work_limit

        self.max_val = length
        crossover = OnePointCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        state = np.random.permutation(self.length)

        if fitness_fn is None:
            fitness_fn = JobScheduling(state, tasks, break_time, work_limit)

        super().__init__(length, fitness_fn, maximize, length, crossover, mutator)
        self.set_state(state)
        #trunc_schedule = fitness_fn.evaluate_order(self.state)
        #self.schedule = trunc_schedule





    def evaluate_population_fitness(self):
        # Calculate fitness
        pop_fitness = []
        for p in self.population:
            member_score = self.fitness_fn.evaluate(p)
            pop_fitness.append(member_score)
        self.pop_fitness = pop_fitness



    def random_pop(self, pop_size):
        """Create a population of random state vectors.

        Parameters
        ----------
        pop_size: int
            Size of population to be created.
        """
        if pop_size <= 0:
            raise Exception("""pop_size must be a positive integer.""")
        elif not isinstance(pop_size, int):
            if pop_size.is_integer():
                pop_size = int(pop_size)
            else:
                raise Exception("""pop_size must be a positive integer.""")


        # Each chromosome represents an order in which tasks should be performed.
        # nd array of pop_size X self.length
        population = np.array([np.random.permutation(self.length) for _ in range(pop_size)])
        self.population = population



        self.evaluate_population_fitness()

    def can_stop(self):
        return int(self.get_fitness()) == int(self.length - 1)
