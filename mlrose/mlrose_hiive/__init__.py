""" MLROSe initialization file."""

# Author: Genevieve Hayes (modified by Andrew Rollings and Qingchuan)
# License: BSD 3 clause

from .algorithms.ga import (genetic_alg)
from .algorithms.sa import (simulated_annealing)
from .algorithms.hc import (hill_climb)
from .algorithms.rhc import (random_hill_climb)
from .algorithms.gd import (gradient_descent)
from .algorithms.mimic import (mimic)
from .algorithms.decay import GeomDecay, ArithDecay, ExpDecay, CustomSchedule
from .algorithms.crossovers import OnePointCrossOver, UniformCrossOver, TSPCrossOver
from .algorithms.mutators import ChangeOneMutator, DiscreteMutator, SwapMutator, ShiftOneMutator
from .fitness import (OneMax, FlipFlop, FourPeaks, SixPeaks, ContinuousPeaks,
                      Knapsack, TravellingSales, Queens, MaxKColor,
                      CustomFitness)
from .neural import NeuralNetwork, LinearRegression, LogisticRegression, _nn_core, NNClassifier
from .neural.activation import (identity, relu,leaky_relu, sigmoid, softmax, tanh)
from .neural.fitness import NetworkWeights
from .neural.utils.weights import (flatten_weights, unflatten_weights)

from .gridsearch import GridSearchMixin

from .opt_probs import DiscreteOpt, ContinuousOpt, KnapsackOpt, TSPOpt, QueensOpt, FlipFlopOpt, MaxKColorOpt, JobSchedulingOpt


from .runners import GARunner, MIMICRunner, RHCRunner, SARunner, NNGSRunner, SKMLPRunner
from .runners import (build_data_filename)
from .generators import (MaxKColorGenerator, QueensGenerator, FlipFlopGenerator, TSPGenerator, KnapsackGenerator, ContinuousPeaksGenerator, FourPeaksGenerator, JobSchedulingGenerator)

from .samples import SyntheticData
from .samples import (plot_synthetic_dataset)

# added by Lyu
from .algorithms.ga_border_check import (genetic_border_alg)
from .algorithms.ga_border_check_front import (genetic_border_front_alg)
from .algorithms.ga_js_border_check_simple import (genetic_js_border_alg_simple)
#from .algorithms.ga_job_scheduling import (genetic_job_scheduling_alg)
from .runners.ga_border_check_runner import GABRunner
from .runners.ga_border_check_front_runner import GABFRunner
from .runners.ga_js_border_check_simple_runner import GAJSRunner
# edits end

