import os
import sys
import pickle
import functions
import numpy as np
import tensorflow as tf

from decimal import *
from data_DE import DataDE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
pre_change = Decimal('0.01')

def obj_function(X):
    return cec_benchmark.Y(X, fun_num)

# Where the program starts
run_num = 50
result_list = []
thresholds = [0.1, 0.15, 0.2]

for threshold in thresholds:
    test_functions = [2, 7, 12, 19, 24]
    fun_list = []
    model_name = "mut_models/mut_NN_t" + str(threshold)
    model = tf.keras.models.load_model(model_name)
    for fun_num in test_functions:
        s = "**** function " + str(fun_num) + ": ****"
        print(s)
        errors = []
        # mutation = np.random.uniform(0.1, 1.9)
        mutation = 0.8
        last_predict = -1
        for run in range(1,run_num + 1):
            print(f"\t# {run} Run: ")
            
            getcontext().prec = 2
            mut_dec = Decimal(str(mutation))
            
            if last_predict == 1:
                mut_dec -= pre_change
            elif last_predict == 2:
                mut_dec += pre_change
            mutation = float(mut_dec)
            
            max_iter = int(max_evals / (popZ size * dim))
            result = differential_evolution(obj_function, bounds=bounds, popsize=popsize,
                                    recombination=recombination, mutation=mutation,
                                    maxiter=max_iter, strategy=strategy, polish=False, tol=0)
            error = result.fun - fDeltas[fun_num - 1]
            getcontext().prec = 100
            error = float(Decimal(str(error)).quantize(Decimal('1.0000')))
            record = (mutation, error)
            print(f"\t\tmutation: {mutation}\n\t\terror: {error}")
            errors.append(record)

            data = DataDE(result, mutation, recombination, popsize)
            X = data.data
            X = X.reshape(1,-1)
            
            std_pkl_path = os.path.join("scalers", "mut", "std.pkl")
            std_reload = pickle.load(open(std_pkl_path,'rb'))
            X = std_reload.transform(X)
            
            pca_pkl_path = os.path.join("scalers", "mut", "pca_" + str(model.input_shape[1]) + ".pkl")
            pca_reload = pickle.load(open(pca_pkl_path,'rb'))
            X = pca_reload.transform(X)

            pred = model.predict(np.array(X))
            last_predict = np.argmax(pred, axis=1)
            last_predict = last_predict[0]
            print(f"\t\tprediction: {last_predict}")

        fun_list.append(errors)

    errors_array = np.array(fun_list)
    print(errors_array)
    with open(f'mut_results_t{threshold}.pickle', 'wb') as f:
        pickle.dump(errors_array, f)

