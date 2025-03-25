""" Classes for defining optimization problem objects."""

import numpy as np

from mlrose_hiive import DiscreteOpt, OneMax


class OneMaxGenerator:
    """Generator class for One Max."""
    @staticmethod
    def generate(seed, size=20):
        np.random.seed(seed)
        fitness = OneMax()
        problem = DiscreteOpt(length=size, fitness_fn=fitness)
        return problem