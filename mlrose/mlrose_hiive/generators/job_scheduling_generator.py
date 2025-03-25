""" Classes for defining optimization problem objects."""

# Author: Qingchuan Lyu
# License: BSD 3 clause

import numpy as np

from mlrose_hiive import JobSchedulingOpt

class Task:
    def __init__(self, task_id, duration, deadline, profit):
        self.task_id = task_id
        self.duration = duration
        self.deadline = deadline
        self.profit = profit

class JobSchedulingGenerator:
    @staticmethod
    def generate(seed, size=20, tasks=[Task('A', 2, 5, 50), Task('B', 3, 8, 70), Task('C', 1, 4, 40), Task('D', 2, 6, 60), Task('E', 4, 10, 100)], break_time=5, work_limit=1):
        np.random.seed(seed)
        problem = JobSchedulingOpt(length=size, tasks=tasks, break_time=break_time, work_limit=work_limit)
        return problem
