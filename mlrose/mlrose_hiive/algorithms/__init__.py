""" Classes for defining algorithms problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

from .ga import (genetic_alg)
from .sa import (simulated_annealing)
from .hc import (hill_climb)
from .rhc import (random_hill_climb)
from .gd import (gradient_descent)
from .mimic import (mimic)

from .crossovers import UniformCrossOver, TSPCrossOver, OnePointCrossOver

from .decay import ArithDecay, CustomSchedule, ExpDecay, GeomDecay
from .mutators import ChangeOneMutator, DiscreteMutator, ShiftOneMutator, SwapMutator

# added by Lyu
from .ga_border_check import (genetic_border_alg)
from .ga_border_check_front import (genetic_border_front_alg)
from .ga_js_border_check_simple import (genetic_js_border_alg_simple)
# edits end

