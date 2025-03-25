import random

# Done
class Task:
    def __init__(self, task_id, duration, deadline, profit):
        self.task_id = task_id
        self.duration = duration
        self.deadline = deadline
        self.profit = profit

# Done
def fitness(schedule, tasks, break_time, work_limit): # maximize profit
    time = 0
    total_profit = 0

    for task_id in schedule:
        task = tasks[task_id]
        if time % work_limit == 0 and time != 0:
            time += break_time
        if time + task.duration <= task.deadline:
            time += task.duration
            total_profit += task.profit
    return total_profit

# Done
def generate_initial_population(pop_size, num_tasks):
    return [random.sample(range(num_tasks), num_tasks) for _ in range(pop_size)]
# Done
def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    child = parent1[:cut] + [task for task in parent2 if task not in parent1[:cut]]
    return child

# Done
def mutate(chromosome):
    i, j = random.sample(range(len(chromosome)), 2)
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

def genetic_algorithm(tasks, break_time, work_limit, population_size=100, generations=100):
    num_tasks = len(tasks)
    population = generate_initial_population(population_size, num_tasks)
    best_schedule = None
    best_score = 0

    for generation in range(generations):
        fitness_scores = [(schedule, fitness(schedule, tasks, break_time, work_limit)) for schedule in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        next_generation = [schedule for schedule, _ in fitness_scores[:population_size // 2]]

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(next_generation, 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:
                mutate(child)
            next_generation.append(child)

        population = next_generation

        best_in_generation = fitness_scores[0]
        if best_in_generation[1] > best_score:
            best_score = best_in_generation[1]
            best_schedule = best_in_generation[0]

    return best_schedule, best_score

# Test data: Task ID, Duration, Deadline, Profit
tasks = [
    Task('A', 2, 5, 50),
    Task('B', 3, 8, 70),
    Task('C', 1, 4, 40),
    Task('D', 2, 6, 60),
    Task('E', 4, 10, 100)
]

break_time = 1  # 1 hour break
work_limit = 4  # 4 hours of continuous work before a break

# Run the genetic algorithm
best_schedule, total_profit = genetic_algorithm(tasks, break_time, work_limit)

# Display results
scheduled_tasks = [tasks[i].task_id for i in best_schedule]
scheduled_tasks, total_profit

""" 
best result
Best Schedule: ['C', 'D', 'B', 'A', 'E']
This sequence represents the order of task execution to maximize profit.

Total Profit: 270
The total profit achieved by scheduling tasks within their deadlines and considering breaks.
"""