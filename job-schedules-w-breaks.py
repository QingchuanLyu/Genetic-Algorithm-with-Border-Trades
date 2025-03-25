import random


# Task class to represent each task
class Task:
    def __init__(self, task_id, duration, deadline):
        self.task_id = task_id
        self.duration = duration
        self.deadline = deadline


# Fitness function: Calculate the number of tasks completed within deadlines
def fitness(schedule, tasks, break_time, work_limit):
    time = 0
    completed_tasks = 0

    for task_id in schedule:
        task = tasks[task_id]
        # Check if we need a break
        if time % work_limit == 0 and time != 0:
            time += break_time
        # Schedule the task if it fits within the deadline
        if time + task.duration <= task.deadline:
            time += task.duration
            completed_tasks += 1
    return completed_tasks


# Generate initial population
def generate_initial_population(pop_size, num_tasks):
    return [random.sample(range(num_tasks), num_tasks) for _ in range(pop_size)]


# Crossover function
def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    child = parent1[:cut] + [task for task in parent2 if task not in parent1[:cut]]
    return child


# Mutation function
def mutate(chromosome):
    i, j = random.sample(range(len(chromosome)), 2)
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]


# Genetic Algorithm for task scheduling with breaks
def genetic_algorithm(tasks, break_time, work_limit, population_size=100, generations=100):
    num_tasks = len(tasks)
    population = generate_initial_population(population_size, num_tasks)
    best_schedule = None
    best_score = 0

    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [(schedule, fitness(schedule, tasks, break_time, work_limit)) for schedule in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Select the top-performing chromosomes
        next_generation = [schedule for schedule, _ in fitness_scores[:population_size // 2]]

        # Generate offspring
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(next_generation, 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # Mutation probability
                mutate(child)
            next_generation.append(child)

        # Update population
        population = next_generation

        # Track the best solution
        best_in_generation = fitness_scores[0]
        if best_in_generation[1] > best_score:
            best_score = best_in_generation[1]
            best_schedule = best_in_generation[0]

    return best_schedule, best_score


# Define tasks: Task ID, Duration, Deadline
tasks = [
    Task('A', 2, 5),
    Task('B', 3, 8),
    Task('C', 1, 4),
    Task('D', 2, 6),
    Task('E', 4, 10)
]

# Parameters: Break time and work limit
break_time = 1  # 1 hour break
work_limit = 4  # 4 hours of continuous work before a break

# Run the genetic algorithm
best_schedule, completed_tasks = genetic_algorithm(tasks, break_time, work_limit)

# Output the result
scheduled_tasks = [tasks[i].task_id for i in best_schedule]
print("Best Schedule:", scheduled_tasks)
print("Number of Tasks Completed:", completed_tasks)
