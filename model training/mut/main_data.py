import sys
import pickle
import functions
import numpy as np

from data_DE import DataDE
from differential_evolution import differential_evolution


fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

dim = 30
max_evals = 10_000 * dim  # like in the CEC'13 benchmark
cec_benchmark = functions.CEC_functions(dim)
bound = 100
bounds = dim * [(bound, -bound)]


def obj_function(X):
    return cec_benchmark.Y(X, fun_num)


runs = 1000
#exp = int(sys.argv[1])
change = 0.1
strategy = 'rand1bin'

for fun_num in range(1, 29):
    runs_list = []
    for run in range(runs):
        recombination = np.random.uniform(0.5, 1)
        popsize = int(np.random.uniform(2, 10))
        mutation = np.random.uniform(0.1, 1.9)
        max_iter = int(max_evals / (popsize * dim))
        result = differential_evolution(obj_function, bounds=bounds, popsize=popsize,
                                        recombination=recombination, mutation=mutation,
                                        maxiter=max_iter, strategy=strategy, polish=False, tol=0)

        print(result.keys())
        break
        data = DataDE(result, mutation, recombination, popsize)
        
        data.change = change
        data.strategy = strategy
        result1 = result
        for mut_change in [-change, change]:
            result = differential_evolution(obj_function, bounds=bounds,
                                            popsize=popsize,
                                            recombination=recombination,
                                            mutation=mutation + mut_change, maxiter=max_iter,
                                            strategy=strategy, polish=False, tol=0)
            data.results.append(result.fun)
            #print(f"Function {fun_num}, Mutation {mutation}, Recombination {recombination}, Popsize {popsize}, result1 {(result1.fun - fDeltas[fun_num - 1]):.2E}, result2: {(result.fun - fDeltas[fun_num - 1]):.2E}")

        runs_list.append(data)
    #pickle.dump(runs_list, open(f"DE_mut_fun{fun_num}_exp{exp}.p", "wb"), protocol=4)

