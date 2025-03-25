""" Functions to implement the Genetic Search algorithms with border swaps.
"""

# Author: Qingchuan Lyu (base code of genetic algorithm was written by Andrew Rollings and Genevieve Hayes in mlrose_hive library)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.decorators import short_name
# added by Lyu
"""
# C1 tuned
def border_swap(problem, next_gen, child, schedule):
    if schedule is None:
        print("order is none.")
        return next_gen
    if len(next_gen) == 0:
        next_gen.append(child)
    else:
        for i in schedule:
            if schedule[i] == -1 and i < len(schedule) - 1:
                schedule[i-1], schedule[i+1] = schedule[i+1], schedule[i-1]
        temp_schedule = [i for i in schedule if i != -1]
        # check if the switched work better
        if problem.eval_fitness(temp_schedule) >= problem.eval_fitness(child):
            next_gen.append(temp_schedule)
        else:
            next_gen.append(child)
    return next_gen

# C2 fine tuned - overtuned - less explore
def border_swap(problem, next_gen, child, schedule):
    if schedule is None:
        print("order is none.")
        return next_gen
    if len(next_gen) == 0:
        next_gen.append(child)
        return next_gen
    else:
        for i in schedule:
            if schedule[i] == -1 and i < len(schedule) - 1:
                schedule[i-1], schedule[i+1] = schedule[i+1], schedule[i-1]
            temp_schedule = [i for i in schedule if i != -1]
            # check if the switched work better
            if problem.eval_fitness(temp_schedule) >= problem.eval_fitness(child):
                next_gen.append(temp_schedule)
                return next_gen
            else:
                schedule[i-1], schedule[i+1] = schedule[i+1], schedule[i-1]
    # if nothing improved
    temp_schedule = [i for i in schedule if i != -1]
    next_gen.append(temp_schedule)
    return next_gen
"""
#B. mutate
def border_swap(problem, next_gen, child, schedule):
    if schedule is None:
        print("order is none.")
        return next_gen
    if len(next_gen) == 0:
        next_gen.append(child)
    else:
        for i in schedule:
            if schedule[i] == -1 and i < len(schedule) - 1:
                schedule[i-1], schedule[i+1] = schedule[i+1], schedule[i-1]
        temp_schedule = [i for i in schedule if i != -1]
        #if problem.eval_fitness(temp_schedule) > problem.eval_fitness(child):
        next_gen.append(temp_schedule)
    return next_gen
"""
# A
#A1. value = profit * deadline / duration
# A2. value = profit
def border_swap(next_gen, child, dic_values):
    if len(next_gen) == 0:
        next_gen.append(child)
    else:
        dic_above = dic_values['above_avg']
        dic_below = dic_values['below_avg']
        new_start = child[0]
        last_start = next_gen[-1][0]
        if ((new_start in dic_below) & (last_start not in dic_below)) or \
                ((new_start in dic_above) & (last_start not in dic_above)):
            next_gen.append(child)
        else:
            child_flip = []
            for i in range(0, len(child), 2):
                j = i + 1
                if j < len(child):
                    #print(child_flip)
                    #print(child[j])
                    #print(child[i])
                    child_flip.extend([child[j], child[i]])
                else:
                    child_flip.append(child[i])
            child_flip = np.array(child_flip[:len(child)])
            #first = child[len(child)//2:]
            #second = child[:len(child)//2]
            #child_flip = np.concatenate([first, second])
            #child_flip = len(child) - 1 - child
            #print("border trades start")
            #print("next gen is", next_gen)
            #print("child is", child)
            #print("second half of child is", child[len(child)//2:])
            #print("first half of child is", child[:len(child)//2])
            #print("child_flip is", child_flip)
            next_gen.append(child_flip)
    return next_gen
"""

# edits end

def _get_hamming_distance_default(population, p1):
    hamming_distances = np.array([np.count_nonzero(p1 != p2) / len(p1) for p2 in population])
    return hamming_distances


def _get_hamming_distance_float(population, p1):
    # use squares instead?
    hamming_distances = np.array([np.abs(np.diff(p1, p2)) / len(p1) for p2 in population])
    return hamming_distances


def _genetic_alg_select_parents(pop_size, problem,
                                get_hamming_distance_func,
                                hamming_factor=0.0):
    mating_probabilities = problem.get_mate_probs()
    if get_hamming_distance_func is not None and hamming_factor > 0.01:
        selected = np.random.choice(pop_size, p=mating_probabilities)
        population = problem.get_population()
        p1 = population[selected]
        hamming_distances = get_hamming_distance_func(population, p1)
        hfa = hamming_factor / (1.0 - hamming_factor)
        hamming_distances = (hamming_distances * hfa) * mating_probabilities
        hamming_distances /= hamming_distances.sum()
        selected = np.random.choice(pop_size, p=hamming_distances)
        p2 = population[selected]

        return p1, p2

    selected = np.random.choice(pop_size,
                                size=2,
                                p=mating_probabilities)
    p1 = problem.get_population()[selected[0]]
    p2 = problem.get_population()[selected[1]]
    return p1, p2


@short_name('ga_js_border_simple')
def genetic_js_border_alg_simple(problem, pop_size=200, pop_breed_percent=0.75, elite_dreg_ratio=0.99,
                minimum_elites=0, minimum_dregs=0, mutation_prob=0.1,
                max_attempts=10, max_iters=np.inf, curve=False, random_state=None,
                state_fitness_callback=None, callback_user_info=None,
                hamming_factor=0.0, hamming_decay_factor=None):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.
    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    pop_breed_percent: float, default 0.75
        Percentage of population to breed in each iteration.
        The remainder of the population will be filled from the elite and
        dregs of the prior generation in a ratio specified by elite_dreg_ratio.
        # elites are individuals from the population with the best fitness values
        # dregs are individuals with the worst fitness values in the population.
    elite_dreg_ratio: float, default:0.95
        The ratio of elites:dregs added directly to the next generation.
        For the default value, 95% of the added population will be elites,
        5% will be dregs.
    minimum_elites: int, default: 0
        Minimum number of elites to be added to next generation
    minimum_dregs: int, default: 0
        Minimum number of dregs to be added to next generation
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    state_fitness_callback: function taking five parameters, default: None
        If specified, this callback will be invoked once per iteration.
        Parameters are (iteration, max attempts reached?, current best state, current best fit, user callback data).
        Return true to continue iterating, or false to stop.
    callback_user_info: any, default: None
        User data passed as last parameter of callback.
    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array of arrays containing the fitness of the entire population
        at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    # how many to breed
    breeding_pop_size = int(pop_size * pop_breed_percent) - (minimum_elites + minimum_dregs)
    # ensure at least one mating.
    if breeding_pop_size < 1:
        raise Exception("""pop_breed_percent must be large enough to ensure at least one mating.""")

    if pop_breed_percent > 1:
        raise Exception("""pop_breed_percent must be less than 1.""")

    if (elite_dreg_ratio < 0) or (elite_dreg_ratio > 1):
        raise Exception("""elite_dreg_ratio must be between 0 and 1.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    fitness_curve = []

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    if state_fitness_callback is not None:
        # initial call with base data
        state_fitness_callback(iteration=0,
                               state=problem.get_state(),
                               fitness=problem.get_adjusted_fitness(),
                               fitness_evaluations=problem.fitness_evaluations,
                               user_data=callback_user_info)

    get_hamming_distance_func = None
    if hamming_factor > 0:
        g1 = problem.get_population()[0][0]
        if isinstance(g1, float) or g1.dtype == 'float64':
            get_hamming_distance_func = _get_hamming_distance_float
        else:
            get_hamming_distance_func = _get_hamming_distance_default

    attempts = 0
    iters = 0

    # initialize survivor count, elite count and dreg count
    survivors_size = pop_size - breeding_pop_size
    dregs_size = max(int(survivors_size * (1.0 - elite_dreg_ratio)) if survivors_size > 1 else 0, minimum_dregs)
    elites_size = max(survivors_size - dregs_size, minimum_elites)
    if dregs_size + elites_size > survivors_size:
        over_population = dregs_size + elites_size - survivors_size
        breeding_pop_size -= over_population

    continue_iterating = True
    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1
        problem.current_iteration += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []
        # added by Lyu
        # A.
        #dic_values = problem.eval_task_value()
        # edits end

        #print(dic_values)
        for _ in range(breeding_pop_size):
            # Select parents
            parent_1, parent_2 = _genetic_alg_select_parents(pop_size=pop_size,
                                                             problem=problem,
                                                             hamming_factor=hamming_factor,
                                                             get_hamming_distance_func=get_hamming_distance_func)

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            # added by Lyu
            # B.
            schedule = problem.eval_order(child)
            next_gen = border_swap(problem, next_gen, child, schedule)
            # A.
            #next_gen = border_swap(next_gen, child, dic_values)

            # edits end


        # fill remaining population with elites/dregs
        if survivors_size > 0:
            last_gen = list(zip(problem.get_population(), problem.get_pop_fitness()))
            sorted_parents = sorted(last_gen, key=lambda f: -f[1])
            best_parents = sorted_parents[:elites_size]
            # added by Lyu
            for p in best_parents:
                curr_child = p[0]
                # B.
                schedule = problem.eval_order(curr_child)
                next_gen = border_swap(problem, next_gen, curr_child, schedule)
                # A.
                #next_gen = border_swap(next_gen, curr_child, dic_values)

            # edits end

            next_gen.extend([p[0] for p in best_parents])
            if dregs_size > 0:
                worst_parents = sorted_parents[-dregs_size:]
                # added by Lyu
                for p in worst_parents:
                    curr_child = p[0]
                    # B.
                    schedule = problem.eval_order(curr_child)
                    next_gen = border_swap(problem, next_gen, curr_child, schedule)
                    # A.
                    #next_gen = border_swap(next_gen, curr_child, dic_values)

                # edits end

        #print("next_gen is ", next_gen)
        #print("pop size is", pop_size)
        next_gen = np.array(next_gen[:pop_size])
        problem.set_population(next_gen) # Change the current population to a specified new population and get the fitness of all members.

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        current_fitness = problem.get_fitness()
        if next_fitness > current_fitness:
            problem.set_state(next_state)
            attempts = 0
        else:
            attempts += 1

        if curve:
            fitness_curve.append((problem.get_adjusted_fitness(), problem.fitness_evaluations))

        # invoke callback
        if state_fitness_callback is not None: # user defined
            max_attempts_reached = (attempts == max_attempts) or (iters == max_iters) or problem.can_stop()
            continue_iterating = state_fitness_callback(iteration=iters,
                                                        attempt=attempts + 1,
                                                        done=max_attempts_reached,
                                                        state=problem.get_state(),
                                                        fitness=problem.get_adjusted_fitness(),
                                                        fitness_evaluations=problem.fitness_evaluations,
                                                        curve=np.asarray(fitness_curve) if curve else None,
                                                        user_data=callback_user_info)

        # decay hamming factor - balance diversity and exploitation; avoid delayed convergence; avoid stuck in local min
        if hamming_decay_factor is not None and hamming_factor > 0.0:
            hamming_factor *= hamming_decay_factor
            hamming_factor = max(min(hamming_factor, 1.0), 0.0)
        # print(hamming_factor)

        # break out if requested
        if not continue_iterating:
            break
    # get_maximize: bool, Whether to maximize the fitness function.
    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()
    return best_state, best_fitness, np.asarray(fitness_curve) if curve else None
