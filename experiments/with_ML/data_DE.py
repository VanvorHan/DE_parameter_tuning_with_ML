import numpy as np


class DataDE:
    def __init__(self, resultsDE, mutation, recombination, popsize):
        self.fun = resultsDE.fun
        self.data = np.zeros((100, 13))
        self.data[:, :11] = resultsDE.data
        self.data[:, 11] = resultsDE.replacements
        self.data[:, 12] = resultsDE.best_updates
        self.mutation = mutation
        self.recombination = recombination
        self.popsize = popsize
        self.results = []

    def print_data(self):
        print(f"Function {self.fun}, Mutation {self.mutation}, Recombination {self.recombination}, Popsize {self.popsize}, Result {self.results}")