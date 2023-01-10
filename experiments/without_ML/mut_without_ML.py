import sys
import pickle
import functions
import numpy as np
from differential_evolution import differential_evolution

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]
## 28 functions

dim = 30
max_evals = 10_000 * dim  # like in the CEC'13 benchmark
cec_benchmark = functions.CEC_functions(dim)

bound = 100
bounds = dim * [(bound, -bound)]
strategy = 'rand1bin'
recombination = 0.9
popsize = 30

def obj_function(X):
    return cec_benchmark.Y(X, fun_num)

# Where the program starts
fun_list = []
test_functions = [2, 7, 12, 19, 24]

for fun_num in test_functions:
    s = "function " + str(fun_num) + ": "
    print(s)
    errors = []
    for run in range(50):
        print(f"\t# {run+1} Run: ")
        mutation = np.random.uniform(0.1, 1.9)
        max_iter = int(max_evals / (popsize * dim))
        result = differential_evolution(obj_function, bounds=bounds, popsize=popsize,
                                    recombination=recombination, mutation=mutation,
                                    maxiter=max_iter, strategy=strategy, polish=False, tol=0)

        error = result.fun - fDeltas[fun_num - 1]
        print(f"\t\terror: {error}")
        errors.append(error)

    fun_list.append(errors)

errors_array = np.array(fun_list)
print(errors_array)
with open('mut_results.pickle', 'wb') as f:
    pickle.dump(errors_array, f)

