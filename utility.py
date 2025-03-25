import itertools
import random

import time

class Task:
    def __init__(self, task_id, duration, deadline, profit):
        self.task_id = task_id
        self.duration = duration
        self.deadline = deadline
        self.profit = profit

def generate_random_tasks(num_tasks, max_duration, max_profit, max_deadline):
    tasks = []
    random.seed(0)
    for i in range(num_tasks):
        task = Task('A' + str(i),
            random.randint(1, max_duration),
            random.randint(1, max_profit),
            random.randint(1, max_deadline))

        tasks.append(task)
    return tasks