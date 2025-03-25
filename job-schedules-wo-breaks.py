import random


# Job class to hold job details
class Job:
    def __init__(self, job_id, deadline, profit):
        self.job_id = job_id
        self.deadline = deadline
        self.profit = profit


# Fitness function: Calculate the total profit of a job schedule
def fitness(schedule, jobs):
    time_slots = [None] * len(jobs)  # Time slots for job scheduling
    total_profit = 0

    # Try to schedule jobs
    for job_id in schedule:
        job = jobs[job_id]
        # Try to place job in a time slot before or at its deadline
        for t in range(job.deadline - 1, -1, -1):
            if time_slots[t] is None:  # Slot is free
                time_slots[t] = job.job_id
                total_profit += job.profit
                break
    return total_profit


# Generate initial population of chromosomes (permutations of jobs)
def generate_initial_population(pop_size, num_jobs):
    return [random.sample(range(num_jobs), num_jobs) for _ in range(pop_size)]


# Crossover function: single-point crossover
def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    child = parent1[:cut] + [job for job in parent2 if job not in parent1[:cut]]
    return child


# Mutation function: swap two jobs in the chromosome
def mutate(chromosome):
    i, j = random.sample(range(len(chromosome)), 2)
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]


# Genetic Algorithm to solve job scheduling problem
def genetic_algorithm(jobs, population_size=100, generations=100):
    num_jobs = len(jobs)
    population = generate_initial_population(population_size, num_jobs)
    best_schedule = None
    best_profit = 0

    for generation in range(generations):
        # Evaluate the fitness of each chromosome (schedule)
        fitness_scores = [(schedule, fitness(schedule, jobs)) for schedule in population]

        # Sort by profit in descending order (best first)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Select the top half of the population for reproduction
        next_generation = [schedule for schedule, _ in fitness_scores[:population_size // 2]]

        # Create new offspring through crossover and mutation
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(next_generation, 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # Mutation probability
                mutate(child)
            next_generation.append(child)

        # Update population for the next generation
        population = next_generation

        # Track the best solution
        best_in_generation = fitness_scores[0]
        if best_in_generation[1] > best_profit:
            best_profit = best_in_generation[1]
            best_schedule = best_in_generation[0]

    # Return the best schedule and profit
    return best_schedule, best_profit


# Define jobs: Job ID, Deadline, and Profit
jobs = [
    Job('A', 2, 100),
    Job('B', 1, 19),
    Job('C', 2, 27),
    Job('D', 1, 25),
    Job('E', 3, 15)
]

# Run the genetic algorithm
best_schedule, best_profit = genetic_algorithm(jobs)

# Output the result
scheduled_jobs = [jobs[i].job_id for i in best_schedule]
print("Best Schedule:", scheduled_jobs)
print("Maximum Profit:", best_profit)
