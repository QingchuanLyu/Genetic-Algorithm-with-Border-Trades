""" Classes for defining optimization problem objects."""

import numpy as np

from mlrose_hiive import DiscreteOpt, FourPeaks


class FourPeaksGenerator:
    """Generator class for Four Peaks."""
    @staticmethod
    def generate(seed, size=20, t_pct=0.1):
        np.random.seed(seed)
        fitness = FourPeaks(t_pct=t_pct)
        problem = DiscreteOpt(length=size, fitness_fn=fitness)
        return problem
