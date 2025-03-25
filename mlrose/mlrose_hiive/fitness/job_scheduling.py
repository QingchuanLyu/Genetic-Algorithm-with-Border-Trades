""" Classes for defining fitness functions."""

# Author: Qingchuan Lyu (based on Genevieve Hayes and Andrew Rollings' mlrose code)
# License: BSD 3 clause

import numpy as np


class JobScheduling:
    """Fitness function for Job Scheduling optimization problem. Evaluates the
    fitness of a state vector :math:`x` as the total profit

    Note
    ----
    The Job Scheduling fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    # added by Lyu
    def __init__(self, schedule=None, tasks=None, break_time=None, work_limit=None):
        if (schedule is None) | (tasks is None) | (break_time is None) | (work_limit is None):
            raise Exception("""schedule, tasks, break_time and work_limit must exist.""")
        self.prob_type = 'discrete'

        self.tasks = tasks
        self.break_time = break_time
        self.work_limit = work_limit

    # grouping tasks by profit / duration
    def group_by_profit_duration(self):
        to_sort = []
        for i, task in enumerate(self.tasks):
            # A1
            #to_sort.append((i, task.profit * task.deadline / task.duration))
            # A2
            to_sort.append((i, task.profit))
        sorted_list = sorted(to_sort, key=lambda x: x[1])
        avg_metric = sum(x[1] for x in sorted_list) / len(self.tasks)
        above_avg = [x[0] for x in sorted_list if x[1] > avg_metric]
        below_avg = [x[0] for x in sorted_list if x[1] <= avg_metric]
        return {'above_avg': above_avg, 'below_avg': below_avg}
    # edits end

    def evaluate(self, schedule):  # , tasks, break_time, work_limit):  # maximize profit
        time = 0
        fitness = 0
        order = []
        for task_id in schedule:
            task = self.tasks[task_id]
            # print("current taks is ", task.task_id)
            # print("expected finish time", time + task.duration)
            # print("deadline", task.deadline)
            expected_time = time + task.duration
            if expected_time <= task.deadline:
                # print("finish a task")
                fitness += task.profit
                order.append(task_id)
                if (task.duration >= self.work_limit) | (expected_time % self.work_limit == 0 and time != 0):
                    # print("take a break")
                    # order.append('-1')
                    time += self.break_time
                time += task.duration
            # print(order)
        return fitness

    # added by Lyu
    def evaluate_order(self, schedule):  # , tasks, break_time, work_limit):  # maximize profit
        #print("input schedule is", schedule)
        time = 0
        fitness = 0
        order = []
        not_done = []
        for task_id in schedule:
            task = self.tasks[task_id]
            #print("current taks is ", task.task_id)
            #rint("expected finish time", time + task.duration)
            #print("deadline", task.deadline)
            expected_time = time + task.duration
            if expected_time <= task.deadline:
                #print("finish a task")
                fitness += task.profit
                order.append(task_id)
                if (task.duration >= self.work_limit) | (expected_time % self.work_limit == 0 and time != 0):
                    #print("take a break")
                    order.append(-1)
                    time += self.break_time
                time += task.duration
            else:
                not_done.append(task_id)
            #print("current order is", order)
            #print("current not done list is", not_done)
        order.extend(not_done)  # append unfinished tasks to the end
        #print("order func returns", order)
        return order
    # edits end

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type
